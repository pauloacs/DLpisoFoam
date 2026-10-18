// Main file for DLpimpleFoam_auto_delta_delta
// Based on DLpisoFoam_auto_delta_delta but adapted for pimpleFoam (incompressible, PIMPLE algorithm)
// Surrogate model is called after the first momentum predictor of every time step.

#include "fvCFD.H"
#include "singlePhaseTransportModel.H"
#include "kinematicMomentumTransportModel.H"
#include "pimpleControl.H"
#include "fvOptions.H"
#include "localEulerDdtScheme.H"
#include "fvcSmooth.H"
#include "POSIX.H"

#include <time.h>
#include <string>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <map>
#include <set>
#include "SurrogateModel.H"
#include "MLSampling.H"
#include "Residuals.H"

template<class Type>
void clearSMStageResiduals(const Foam::polyMesh& mesh)
{
    const Foam::List<Foam::word> fieldNames =
        Foam::Residuals<Type>::fieldNames(mesh);

    forAll(fieldNames, fieldi)
    {
        const_cast
        <
            Foam::DynamicList<Foam::SolverPerformance<Type>>&
        >
        (
            Foam::Residuals<Type>::field(mesh, fieldNames[fieldi])
        ).clear();
    }
}

template<class Type>
Foam::label smResidualCount(const Foam::polyMesh& mesh, const Foam::word& fieldName)
{
    return Foam::Residuals<Type>::found(mesh, fieldName)
        ? Foam::Residuals<Type>::field(mesh, fieldName).size() : 0;
}

template<class Type>
Foam::scalar smFirstResidualSince
(
    const Foam::polyMesh& mesh,
    const Foam::word& fieldName,
    const Foam::label firstSolve
)
{
    if (!Foam::Residuals<Type>::found(mesh, fieldName)) return Foam::GREAT;
    const auto& residuals = Foam::Residuals<Type>::field(mesh, fieldName);
    return firstSolve < residuals.size()
        ? Foam::cmptMax(residuals[firstSolve].initialResidual()) : Foam::GREAT;
}

// Macro to bake in source directory at compile time via -DSOLVER_SOURCE_DIR
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    // create argument list
    Foam::argList args(argc, argv, true, true, /*initialise=*/false);
    if (!args.checkRootCase())
    {
        Foam::FatalError.exit();
    }

    #include "postProcess.H"

    #include "createTime.H"
    #include "createMesh.H"
    #include "createDyMControls.H"
    #include "createFields.H"
    #include "initContinuityErrs.H"

    turbulence->validate();

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    Info<< "\nStarting time loop\n" << endl;

    // Read runtime behavior from python_module.py
    bool train = true;      // default
    bool predictDP = false; // default var_to_predict='ddp'
    bool ddpFullCoarsePrediction = false; // default ddp_full_coarse_prediction=False
    bool predictDDUCorrOutput = false;
    bool predictDUOutput = false;
    bool verboseSM = true;  // default matches python_module.py's verbose=True
    std::map<std::string, bool> pythonInputFlags;
    bool pythonModuleRead = false;
    {
        auto trim = [](const std::string& s) -> std::string
        {
            const std::string ws = " \t\r\n";
            const std::size_t beg = s.find_first_not_of(ws);
            if (beg == std::string::npos)
            {
                return std::string();
            }
            const std::size_t end = s.find_last_not_of(ws);
            return s.substr(beg, end - beg + 1);
        };

        // Each key is only consumed from its FIRST matching assignment line.
        // python_module.py also contains kwarg-forwarding lines further down
        // (e.g. "var_to_predict=var_to_predict," inside the load_tucker_and_NN
        // call), which textually match "key == ...", but are NOT literal
        // assignments and must not overwrite the value already parsed from
        // the real "var_to_predict = 'dp'"/"train = True" lines above them.
        bool trainParsed = false;
        bool varToPredictParsed = false;
        bool verboseParsed = false;
        bool ddpFullCoarsePredictionParsed = false;
        bool predictDDUCorrOutputParsed = false;
        bool predictDUOutputParsed = false;

        std::ifstream pythonModuleFile("python_module.py");
        if (pythonModuleFile.good())
        {
            pythonModuleRead = true;
            std::string line;
            while (std::getline(pythonModuleFile, line))
            {
                const std::size_t commentPos = line.find('#');
                const std::string cleanLine = trim(
                    commentPos == std::string::npos ? line : line.substr(0, commentPos)
                );

                if (cleanLine.empty())
                {
                    continue;
                }

                const std::size_t eqPos = cleanLine.find('=');
                if (eqPos == std::string::npos)
                {
                    continue;
                }

                std::string key = trim(cleanLine.substr(0, eqPos));
                std::string value = trim(cleanLine.substr(eqPos + 1));

                std::transform(
                    key.begin(),
                    key.end(),
                    key.begin(),
                    [](unsigned char c) { return static_cast<char>(std::tolower(c)); }
                );
                std::transform(
                    value.begin(),
                    value.end(),
                    value.begin(),
                    [](unsigned char c) { return static_cast<char>(std::tolower(c)); }
                );

                // First literal assignment wins; kwarg-forwarding lines are not booleans.
                if
                (
                    key.size() > 6
                 && key.compare(key.size() - 6, 6, "_input") == 0
                 && (value == "true" || value == "false")
                 && pythonInputFlags.find(key) == pythonInputFlags.end()
                )
                {
                    pythonInputFlags[key] = (value == "true");
                    continue;
                }

                // Read train = True/False
                if (key == "train")
                {
                    if (trainParsed)
                    {
                        continue;
                    }
                    if (value.find("false") != std::string::npos)
                    {
                        train = false;
                        trainParsed = true;
                        Foam::Info<< "  [DLpimpleFoam] Read train=False from python_module.py - sampling disabled" << Foam::nl;
                    }
                    else if (value.find("true") != std::string::npos)
                    {
                        train = true;
                        trainParsed = true;
                        Foam::Info<< "  [DLpimpleFoam] Read train=True from python_module.py - sampling enabled" << Foam::nl;
                    }
                    continue;
                }

                // Read var_to_predict = 'ddp' or 'dp'
                if (key == "var_to_predict")
                {
                    if (varToPredictParsed)
                    {
                        continue;
                    }
                    if (value.find("ddp") != std::string::npos)
                    {
                        predictDP = false;
                        varToPredictParsed = true;
                        Foam::Info<< "  [DLpimpleFoam] Read var_to_predict='ddp' from python_module.py" << Foam::nl;
                    }
                    else if (value.find("dp") != std::string::npos)
                    {
                        predictDP = true;
                        varToPredictParsed = true;
                        Foam::Info<< "  [DLpimpleFoam] Read var_to_predict='dp' from python_module.py" << Foam::nl;
                    }
                    else
                    {
                        predictDP = false;
                        varToPredictParsed = true;
                        Foam::Info<< "  [DLpimpleFoam] WARNING: Unknown var_to_predict value '" << value
                                  << "' in python_module.py. Falling back to 'ddp'." << Foam::nl;
                    }
                    continue;
                }

                // Read verbose = True/False (controls SM diagnostics printing)
                if (key == "verbose")
                {
                    if (verboseParsed)
                    {
                        continue;
                    }
                    if (value.find("false") != std::string::npos)
                    {
                        verboseSM = false;
                        verboseParsed = true;
                        Foam::Info<< "  [DLpimpleFoam] Read verbose=False from python_module.py - SM diagnostics printing disabled" << Foam::nl;
                    }
                    else if (value.find("true") != std::string::npos)
                    {
                        verboseSM = true;
                        verboseParsed = true;
                        Foam::Info<< "  [DLpimpleFoam] Read verbose=True from python_module.py - SM diagnostics printing enabled" << Foam::nl;
                    }
                    continue;
                }

                // Read ddp_full_coarse_prediction = True/False (ddp-mode combination mode)
                if (key == "ddp_full_coarse_prediction")
                {
                    if (ddpFullCoarsePredictionParsed)
                    {
                        continue;
                    }
                    if (value.find("false") != std::string::npos)
                    {
                        ddpFullCoarsePrediction = false;
                        ddpFullCoarsePredictionParsed = true;
                        Foam::Info<< "  [DLpimpleFoam] Read ddp_full_coarse_prediction=False from python_module.py" << Foam::nl;
                    }
                    else if (value.find("true") != std::string::npos)
                    {
                        ddpFullCoarsePrediction = true;
                        ddpFullCoarsePredictionParsed = true;
                        Foam::Info<< "  [DLpimpleFoam] Read ddp_full_coarse_prediction=True from python_module.py" << Foam::nl;
                    }
                    continue;
                }

                if (key == "predict_dducorr_output" && !predictDDUCorrOutputParsed)
                {
                    predictDDUCorrOutput = (value.find("true") != std::string::npos);
                    predictDDUCorrOutputParsed = true;
                }
                if (key == "predict_du_output" && !predictDUOutputParsed)
                {
                    predictDUOutput = (value.find("true") != std::string::npos);
                    predictDUOutputParsed = true;
                }

            }
            pythonModuleFile.close();
        }
    }

    // Unreadable python_module.py: compute everything; a missing flag means disabled.
    const auto inputWanted = [&](const char* key) -> bool
    {
        if (!pythonModuleRead)
        {
            return true;
        }
        const auto flag = pythonInputFlags.find(key);
        return flag != pythonInputFlags.end() && flag->second;
    };
    const bool wantDivUStar = inputWanted("add_divustar_input");
    const bool wantLaplaceDpPrev = inputWanted("add_laplacian_dpprev_input");
    const bool wantUDotGradDpPrev = inputWanted("add_udotgraddpprev_input");
    const bool wantGradDpPrevMag = inputWanted("add_graddpprevmag_input");
    const bool wantRAUGradDpPrev = inputWanted("add_raugraddpprev_input");
    const bool wantDivRAUGradDpPrev = inputWanted("add_divraugraddpprev_input");
    const bool wantResidual = inputWanted("add_pressureeqresidualp_input");
    const bool wantRAUGradpPrev = inputWanted("add_raugradpprev_input");
    const bool wantDivRAUGradpPrev = inputWanted("add_divraugradpprev_input");
    const bool wantGradDpPrevInput = inputWanted("add_graddpprev_input");
    const bool wantHbyAInput = inputWanted("include_hbya_input");
    const bool wantDivHbyAInput = inputWanted("include_divhbya_input");
    const bool wantDHbyA = inputWanted("include_dhbya_input");
    const bool wantDDivHbyA = inputWanted("include_ddivhbya_input");
    const bool wantRAUInput = inputWanted("include_rau_input");

    const bool needDivHbyA = wantDivHbyAInput || wantDDivHbyA || wantResidual;
    const bool needHbyA = wantHbyAInput || wantDHbyA || needDivHbyA;
    const bool needRAU = wantRAUInput || needHbyA || wantRAUGradDpPrev
        || wantDivRAUGradDpPrev || wantResidual || wantRAUGradpPrev
        || wantDivRAUGradpPrev;
    const bool needGradDp = (wantGradDpPrevInput && !predictDP)
        || wantGradDpPrevMag || wantUDotGradDpPrev || wantRAUGradDpPrev;
    const bool needGradpPrev = (wantGradDpPrevInput && predictDP)
        || wantRAUGradpPrev;

    Info<< "[SM FEATURES] computing:"
        << " rAU=" << needRAU << " HbyA=" << needHbyA
        << " divHbyA=" << needDivHbyA << " residual=" << wantResidual
        << " gradDp=" << needGradDp << " gradpPrev=" << needGradpPrev
        << " laplaceDp=" << wantLaplaceDpPrev << " divUStar=" << wantDivUStar
        << " dHbyA=" << wantDHbyA << " dDivHbyA=" << wantDDivHbyA << nl;

    // Three cases for how the SM's pressure prediction is produced/reported:
    //   a) predictDP=true                      (var_to_predict='dp')
    //   b) predictDP=false, ddpFullCoarsePrediction=true  (ddp mode, full coarse)
    //   c) predictDP=false, ddpFullCoarsePrediction=false (ddp mode, residual correction)
    //
    // fullDpReconstruction is true for (a) and (b): pSM is already a full
    // dp-space estimate (p_prev + dpML) rather than a ddp-residual correction
    // added on top of pExtrap (pSM = pExtrap + ddpML, case (c) only).
    const bool fullDpReconstruction = predictDP || ddpFullCoarsePrediction;

    // writeDdpFields is true ONLY for case (c): that is the only case where
    // the native ddp-space fields (ddp, ddpML, and the ddp_SM/ddp_truth delta
    // fields below) are meaningful for debugging. In (a) there is no ddp
    // involved at all, and in (b) ddp is folded away into the coarse dp
    // reconstruction, so ddp/ddpML no longer represent a meaningful quantity
    // on their own.
    const bool writeDdpFields = !fullDpReconstruction;

    // writeNativeDeltaFields controls smDeltaField/truthDeltaField below (the
    // var_to_predict-named delta fields: "dp_SM"/"dp_truth" in dp mode, or
    // "ddp_SM"/"ddp_truth" in ddp mode). Always written in (a) and (c);
    // suppressed only in (b), where they would just be ddp-space duplicates —
    // the dp_SM/dp_truth fields further below already cover that case.
    const bool writeNativeDeltaFields = predictDP || writeDdpFields;

    ddp.writeOpt() = writeDdpFields ? IOobject::AUTO_WRITE : IOobject::NO_WRITE;
    ddpML.writeOpt() = writeDdpFields ? IOobject::AUTO_WRITE : IOobject::NO_WRITE;

    // Surrogate model (encapsulates all Python/NumPy state)
    SurrogateModel surrogate
    (
        mesh, p, U,
        dUStar, dUStarPrev,
        dpML, dp, dpPrev, p_prev,
        ddUStar, ddUStarPrev,
        ddpML, ddpPrev, ddp,
        grad_dpPrev,
        grad_pPrev,
        laplace_dpPrev,
        uDotGradDpPrev,
        gradDpPrevMag,
        ddUStarML,
        divUFirstPred,
        dUCorrPrev, ddUCorrPrev,
        rAU_ML, HbyA_ML, divHbyA_ML, dHbyA_ML, dDivHbyA_ML,
        rAUGradDpPrev_ML, divRAUGradDpPrev_ML, pressureEqResidualp_ML,
        rAUGradpPrev_ML, divRAUGradpPrev_ML,
        predictDP,
        ddpFullCoarsePrediction
    );
    bool surrogateActive = false;

    // Source script directory is baked in at compile time via -DSOLVER_SOURCE_DIR
    std::string sourceScriptDir(SOLVER_SOURCE_DIR);
    Info<< "Source script directory: " << sourceScriptDir << nl << endl;

    // Read ML sampling parameters from system/MLSamplingDict
    IOdictionary mlDict
    (
        IOobject
        (
            "MLSamplingDict",
            runTime.system(),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        )
    );
    int warmUpSteps     = mlDict.lookupOrDefault<int>("warmUpSteps",     10);
    int burstSteps      = mlDict.lookupOrDefault<int>("burstSteps",       10);
    int burstInterval   = mlDict.lookupOrDefault<int>("burstInterval",     1);
    int regularInterval = mlDict.lookupOrDefault<int>("regularInterval",   5);
    int retrainInterval       = mlDict.lookupOrDefault<int>("retrainInterval",       20);
    int windowFrames          = mlDict.lookupOrDefault<int>("windowFrames",           30);
    int waitBeforeResampling  = mlDict.lookupOrDefault<int>("waitBeforeResampling",    0);
    Switch applySMPressureConsistencyCorrection =
        mlDict.lookupOrDefault<Switch>("applySMPressureConsistencyCorrection", true);
    Switch applySMPredictedUCorrection =
        mlDict.lookupOrDefault<Switch>("applySMPredictedUCorrection", true);

    Switch postSMInjectionReconstruction =
        mlDict.lookupOrDefault<Switch>("postSMInjectionReconstruction", false);
    Switch smPressureForceThenReconstruct =
        mlDict.lookupOrDefault<Switch>("smPressureForceThenReconstruct", false);
    Switch smOracleCompatibleIntegration =
        mlDict.lookupOrDefault<Switch>("smOracleCompatibleIntegration", false);
    const scalar smOuterUResidualTolerance = mlDict.lookupOrDefault<scalar>("smOuterUResidualTolerance", 5e-4);
    const scalar smOuterPResidualTolerance = mlDict.lookupOrDefault<scalar>("smOuterPResidualTolerance", 5e-4);
    const bool smAdaptivePressureCorrectors = mlDict.lookupOrDefault<bool>("smAdaptivePressureCorrectors", true);
    const label smMinPressureCorrectors = mlDict.lookupOrDefault<label>("smMinPressureCorrectors", 1);
    const scalar smPressureCorrectorTolerance = mlDict.lookupOrDefault<scalar>("smPressureCorrectorTolerance", 1e-5);
    Switch smUseInputPredictorState =
        mlDict.lookupOrDefault<Switch>("smUseInputPredictorState", true);
    Switch smUseVelocityExtrapolation =
        mlDict.lookupOrDefault<Switch>("smUseVelocityExtrapolation", false);
    const scalar smVelocityExtrapolationAlpha =
        mlDict.lookupOrDefault<scalar>("smVelocityExtrapolationAlpha", 1.0);
    label smStateReconstructionCorrectors =
        mlDict.lookupOrDefault<label>("smStateReconstructionCorrectors", 1);
    Switch smAdaptiveStateReconstruction =
        mlDict.lookupOrDefault<Switch>("smAdaptiveStateReconstruction", true);
    label smMinStateReconstructionCorrectors =
        mlDict.lookupOrDefault<label>("smMinStateReconstructionCorrectors", 1);
    scalar smStateReconstructionResidualTolerance =
        mlDict.lookupOrDefault<scalar>
        (
            "smStateReconstructionResidualTolerance",
            1e-4
        );
    scalar smStateReconstructionEquationRelaxation =
        mlDict.lookupOrDefault<scalar>
        (
            "smStateReconstructionEquationRelaxation",
            1.0
        );
    Switch smAitkenStateReconstruction =
        mlDict.lookupOrDefault<Switch>("smAitkenStateReconstruction", true);
    scalar smAitkenMinFactor =
        mlDict.lookupOrDefault<scalar>("smAitkenMinFactor", 0.2);
    scalar smAitkenMaxFactor =
        mlDict.lookupOrDefault<scalar>("smAitkenMaxFactor", 1.8);

    if (smOracleCompatibleIntegration
        && (LTS || smUseInputPredictorState || !postSMInjectionReconstruction
            || smOuterUResidualTolerance <= 0 || smOuterPResidualTolerance <= 0
            || smMinPressureCorrectors < 1 || smPressureCorrectorTolerance <= 0))
    {
        FatalErrorInFunction
            << "Oracle-compatible integration requires transient timestepping, restored input state, "
            << "post-injection reconstruction and positive stopping tolerances."
            << exit(FatalError);
    }
    if (!smUseInputPredictorState && !postSMInjectionReconstruction)
    {
        FatalErrorInFunction
            << "smUseInputPredictorState=false requires "
            << "postSMInjectionReconstruction=true." << exit(FatalError);
    }
    if (!std::isfinite(smVelocityExtrapolationAlpha) || smVelocityExtrapolationAlpha < 0
        || (smUseVelocityExtrapolation
            && (LTS || predictDUOutput || predictDDUCorrOutput || smUseInputPredictorState || !postSMInjectionReconstruction)))
    {
        FatalErrorInFunction
            << "Velocity extrapolation requires a finite nonnegative alpha, transient pressure-only SM, "
            << "smUseInputPredictorState=false and postSMInjectionReconstruction=true."
            << exit(FatalError);
    }
    if (predictDUOutput && (predictDDUCorrOutput || smUseInputPredictorState || !postSMInjectionReconstruction))
    {
        FatalErrorInFunction
            << "predict_dU_output requires predict_ddUCorr_output=False, "
            << "smUseInputPredictorState=false and postSMInjectionReconstruction=true."
            << exit(FatalError);
    }
    if (smStateReconstructionCorrectors < 1)
    {
        FatalErrorInFunction
            << "smStateReconstructionCorrectors must be at least 1."
            << exit(FatalError);
    }
    if
    (
        smMinStateReconstructionCorrectors < 1
     || smMinStateReconstructionCorrectors > smStateReconstructionCorrectors
    )
    {
        FatalErrorInFunction
            << "smMinStateReconstructionCorrectors must be between 1 and "
            << "smStateReconstructionCorrectors." << exit(FatalError);
    }
    if
    (
        smStateReconstructionResidualTolerance <= 0
     || smStateReconstructionEquationRelaxation <= 0
     || smStateReconstructionEquationRelaxation > 1
     || smAitkenMinFactor <= 0
     || smAitkenMaxFactor < smAitkenMinFactor
    )
    {
        FatalErrorInFunction
            << "Invalid SM reconstruction tolerances/relaxation/Aitken bounds."
            << exit(FatalError);
    }

    Info<< "SM post-injection reconstruction="
        << postSMInjectionReconstruction
        << " predictVelocity=" << predictDUOutput
        << " velocityExtrapolation=" << smUseVelocityExtrapolation
        << " velocityExtrapolationAlpha=" << smVelocityExtrapolationAlpha
        << " pressureForceThenReconstruct=" << smPressureForceThenReconstruct
        << " oracleCompatible=" << smOracleCompatibleIntegration
        << " equationRelaxationSource=" << (smOracleCompatibleIntegration ? "fvSolution" : "MLSamplingDict")
        << " useInputPredictorState=" << smUseInputPredictorState
        << " correctors=" << smStateReconstructionCorrectors
        << " adaptive=" << smAdaptiveStateReconstruction
        << " minCorrectors=" << smMinStateReconstructionCorrectors
        << " residualTolerance=" << smStateReconstructionResidualTolerance
        << " equationRelaxation=" << smStateReconstructionEquationRelaxation
        << " aitken=" << smAitkenStateReconstruction
        << " aitkenBounds=(" << smAitkenMinFactor
        << ' ' << smAitkenMaxFactor << ')' << nl;

    std::set<std::string> sampleFields;
    if (pythonModuleRead && !pythonInputFlags.empty())
    {
        sampleFields.insert("ddp");
        const std::pair<const char*, const char*> inputDatasetMap[] =
        {
            {"add_u_input", "U"},
            {"add_dustar_input", "dUStar"},
            {"add_duprev_input", "dUPrev"},
            {"add_ddustar_input", "ddUStar"},
            {"add_ddustardiff_input", "ddUStarDiff"},
            {"add_dpprev_input", "dpPrev"},
            {"add_pprev_input", "p_prev"},
            {"add_ddpprev_input", "ddpPrev"},
            {"add_laplacian_dpprev_input", "laplaceDpPrev"},
            {"add_udotgraddpprev_input", "uDotGradDpPrev"},
            {"add_graddpprevmag_input", "gradDpPrevMag"},
            {"include_rau_input", "rAU"},
            {"include_hbya_input", "HbyA"},
            {"include_divhbya_input", "divHbyA"},
            {"include_dhbya_input", "dHbyA"},
            {"include_ddivhbya_input", "dDivHbyA"},
            {"add_raugraddpprev_input", "rAUGradDpPrev"},
            {"add_divraugraddpprev_input", "divRAUGradDpPrev"},
            {"add_pressureeqresidualp_input", "pressureEqResidualp"},
            {"add_raugradpprev_input", "rAUGradpPrev"},
            {"add_divraugradpprev_input", "divRAUGradpPrev"},
            {"add_divustar_input", "divUFirstPred"},
            {"add_ducorrprev_input", "dUCorrPrev"},
            {"add_dducorrprev_input", "ddUCorrPrev"}
        };
        for (const auto& entry : inputDatasetMap)
        {
            if (inputWanted(entry.first))
            {
                sampleFields.insert(entry.second);
            }
        }
        if (inputWanted("add_graddpprev_input"))
        {
            sampleFields.insert(predictDP ? "gradpPrev" : "gradDpPrev");
        }
        if (inputWanted("add_udotnwall_input"))
        {
            sampleFields.insert("U");
        }
        if (predictDP)
        {
            sampleFields.insert("dp");
            sampleFields.insert("dpPrev");
        }
        if (predictDDUCorrOutput)
        {
            sampleFields.insert("ddUCorr");
        }
        if (predictDUOutput)
        {
            sampleFields.insert("dU");
            sampleFields.insert("dUPrev");
        }
        Info<< "[DataSampler] Selected HDF5 field count: "
            << sampleFields.size() << Foam::nl;
    }

    DataSampler dataSampler(
        mesh, UStar, dUStar, dp, ddUStar, ddUStarPrev, ddp, dpPrev, ddpPrev, grad_dpPrev, grad_pPrev, laplace_dpPrev, uDotGradDpPrev, gradDpPrevMag, p_prev, divUFirstPred, ddUCorr, dUCorrPrev, ddUCorrPrev, rAU_ML, HbyA_ML, divHbyA_ML, dHbyA_ML, dDivHbyA_ML, rAUGradDpPrev_ML, divRAUGradDpPrev_ML, pressureEqResidualp_ML, rAUGradpPrev_ML, divRAUGradpPrev_ML, "ML_data",
        sourceScriptDir,
        train,
        warmUpSteps, burstSteps, burstInterval, regularInterval, retrainInterval, windowFrames,
        waitBeforeResampling,
        sampleFields
    );

    // SM diagnostics snapshots and cumulative improvement statistics.
    volScalarField pExtrap
    (
        IOobject
        (
            "pExtrap",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        p
    );
    volScalarField pSM
    (
        IOobject
        (
            "pSM",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        p
    );

    const word smDeltaFieldName = predictDP ? word("dp_SM") : word("ddp_SM");
    const word truthDeltaFieldName = predictDP ? word("dp_truth") : word("ddp_truth");

    volScalarField smDeltaField
    (
        IOobject
        (
            smDeltaFieldName,
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            writeNativeDeltaFields ? IOobject::AUTO_WRITE : IOobject::NO_WRITE
        ),
        p
    );

    volScalarField truthDeltaField
    (
        IOobject
        (
            truthDeltaFieldName,
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            writeNativeDeltaFields ? IOobject::AUTO_WRITE : IOobject::NO_WRITE
        ),
        p
    );

    // ddp mode only: smDeltaField/truthDeltaField above are named "ddp_SM"/
    // "ddp_truth", but the quantity that ultimately matters is dp, not ddp.
    // These extra fields always hold the dp-space SM prediction/truth so it
    // can be inspected directly, regardless of var_to_predict. In dp mode,
    // smDeltaField/truthDeltaField already ARE "dp_SM"/"dp_truth", so these
    // are left unset (nullptr) to avoid registering a duplicate field name.
    autoPtr<volScalarField> dpSMFieldPtr;
    autoPtr<volScalarField> dpTruthFieldPtr;
    if (!predictDP)
    {
        dpSMFieldPtr.set
        (
            new volScalarField
            (
                IOobject
                (
                    "dp_SM",
                    runTime.timeName(),
                    mesh,
                    IOobject::NO_READ,
                    IOobject::AUTO_WRITE
                ),
                p
            )
        );
        dpTruthFieldPtr.set
        (
            new volScalarField
            (
                IOobject
                (
                    "dp_truth",
                    runTime.timeName(),
                    mesh,
                    IOobject::NO_READ,
                    IOobject::AUTO_WRITE
                ),
                p
            )
        );
    }

    volVectorField jointDUSM
    (
        IOobject("dU_SM", runTime.timeName(), mesh, IOobject::NO_READ,
            (predictDUOutput || smUseVelocityExtrapolation) ? IOobject::AUTO_WRITE : IOobject::NO_WRITE),
        U - U
    );
    volVectorField jointDUTruth
    (
        IOobject("dU_truth", runTime.timeName(), mesh, IOobject::NO_READ,
            (predictDUOutput || smUseVelocityExtrapolation) ? IOobject::AUTO_WRITE : IOobject::NO_WRITE),
        U - U
    );
    bool smDiagnosticsActiveThisStep = false;
    bool smOperatorDiagnosticsDoneThisStep = false;
    bool smFirstGamgResidualCapturedThisStep = false;
    label smDiagnosticsSamples = 0;
    scalar cumulativePressureRMSImprovement = 0.0;
    scalar cumulativeGradientRMSImprovement = 0.0;
    scalar cumulativeLaplacianRMSImprovement = 0.0;
    scalar cumulativeOperatorRMSImprovement = 0.0;
    scalar cumulativeContinuityRMSImprovement = 0.0;

    // End-of-timestep SM-vs-SM comparisons against BOTH naive baselines
    // (p_prev only, and p_prev + dpPrev trend extrapolation), independent
    // of the active var_to_predict mode, so it is possible to confirm the
    // SM beats both simple heuristics, not just the one used as pExtrap.
    scalar cumulativePressureRMSImprovementVsPrevOnly = 0.0;
    scalar cumulativePressureRMSImprovementVsTrend = 0.0;
    scalar cumulativeGradientRMSImprovementVsPrevOnly = 0.0;
    scalar cumulativeGradientRMSImprovementVsTrend = 0.0;
    scalar cumulativeLaplacianRMSImprovementVsPrevOnly = 0.0;
    scalar cumulativeLaplacianRMSImprovementVsTrend = 0.0;

    scalar smOperatorRMSImprovementThisStep = 0.0;
    scalar smContinuityRMSImprovementThisStep = 0.0;
    scalar smOperatorRmsExtrapThisStep = 0.0;
    scalar smOperatorRmsSMThisStep = 0.0;
    scalar smFirstGamgInitialResidualThisStep = 0.0;

    bool pimpleSMOverridesApplied = false;

    auto applyPimpleSMOverrides = [&]()
    {
        if (pimpleSMOverridesApplied)
        {
            return;
        }

        const dictionary& fvSolDict = mesh.solutionDict();

        if (!fvSolDict.found("PIMPLE_SM"))
        {
            return;
        }

        Info<< "Applying PIMPLE_SM solver settings for surrogate-accelerated phase." << nl;

        const dictionary& pimpleSMDict = fvSolDict.subDict("PIMPLE_SM");

        if (fvSolDict.found("PIMPLE"))
        {
            dictionary& pimpleDict =
                const_cast<dictionary&>(fvSolDict.subDict("PIMPLE"));

            forAllConstIter(dictionary, pimpleSMDict, iter)
            {
                pimpleDict.set(iter().clone(pimpleDict).ptr());
                Info<< "  Override: " << iter().keyword() << nl;
            }

            pimple.read();
            pimpleSMOverridesApplied = true;
        }
        else
        {
            FatalErrorInFunction
                << "PIMPLE_SM found but PIMPLE dict not found in fvSolution"
                << exit(FatalError);
        }
    };

    // SM diagnostics helper lambdas (computeScalarErrorMetrics,
    // computeVectorErrorMagnitudeMetrics, computeImprovementPercent,
    // printMetricAssessment, printMetricAssessment3, helpedState, signedStr).
    // Captured by reference, shared by pEqn.H and smEndOfStepDiagnostics.H.
    #include "smDiagnosticsHelpers.H"

    pimpleControl& persistentPimple = pimple;
    if (smOracleCompatibleIntegration)
    {
        persistentPimple.checkOut();
    }
    bool smHaveVelocityHistory = false;
    scalar smPreviousCompletedDeltaT = 0.0;
    while (runTime.loop())
    {
        autoPtr<pimpleControl> stepPimple;
        if (smOracleCompatibleIntegration)
        {
            stepPimple.reset(new pimpleControl(mesh));
            clearSMStageResiduals<vector>(mesh);
            clearSMStageResiduals<scalar>(mesh);
        }
        pimpleControl& pimple = smOracleCompatibleIntegration ? stepPimple() : persistentPimple;
        const auto stepWallStart = std::chrono::steady_clock::now();
        scalar stepUEqnWall = 0.0;
        scalar stepPEqnWall = 0.0;
        scalar stepTurbulenceWall = 0.0;
        scalar stepSMWall = 0.0;
        scalar stepReconstructionWall = 0.0;
        Info<< "Time = " << runTime.timeName() << nl << endl;
        smDiagnosticsActiveThisStep = false;
        smOperatorDiagnosticsDoneThisStep = false;
        smFirstGamgResidualCapturedThisStep = false;
        smOperatorRMSImprovementThisStep = 0.0;
        smContinuityRMSImprovementThisStep = 0.0;
        smOperatorRmsExtrapThisStep = 0.0;
        smOperatorRmsSMThisStep = 0.0;
        smFirstGamgInitialResidualThisStep = 0.0;

        if (LTS)
        {
            #include "setRDeltaT.H"
        }
        else
        {
            #include "CourantNo.H"
            #include "setDeltaT.H"
        }

        // --- Pressure-velocity PIMPLE corrector loop
        const auto pimpleWallStart = std::chrono::steady_clock::now();
        bool smInputOnlyStage = surrogateActive && !smUseInputPredictorState;
        bool smInputOnlyStageDone = false;
        autoPtr<surfaceScalarField> smInputPhiStart;
        while (smInputOnlyStage || pimple.loop())
        {
            const label smFirstUSolve = smResidualCount<vector>(mesh, U.name());
            int counter = 0;
            const bool firstSMInputStage = smInputOnlyStage
                || (!smInputOnlyStageDone && pimple.firstPimpleIter());

            // Store previous time step values once, before the first input predictor.
            if (firstSMInputStage)
            {
                p_prev = p;
                U_time_prev = U;
                dUStarPrev = dUStar;
                dUCorrPrev = dUCorr;
                ddUStarPrev = ddUStar;
                ddUCorrPrev = ddUCorr;
                dpPrev = dp;
                ddpPrev = ddp;
                if (needGradDp)
                {
                    grad_dpPrev = grad_dp;
                }

                if (wantDHbyA)
                {
                    HbyA_ML_prev = HbyA_ML;
                }
                if (wantDDivHbyA)
                {
                    divHbyA_ML_prev = divHbyA_ML;
                }
            }


            if (firstSMInputStage)
            {
                if (smInputOnlyStage)
                {
                    smInputPhiStart.reset
                    (
                        new surfaceScalarField
                        (
                            IOobject
                            (
                                "smInputPhiStart", runTime.timeName(), mesh,
                                IOobject::NO_READ, IOobject::NO_WRITE, false
                            ),
                            phi
                        )
                    );
                    Info<< "[SM INPUT] feature-only UEqn before normal PIMPLE"
                        << nl;
                }
                // Initial pressure guess before UEqn/surrogate correction.
                // In 'ddp' mode the SM corrects a linear (trend) extrapolation,
                // so pExtrap = p_prev + dpPrev is the right baseline/initial guess.
                // In 'dp' mode the SM predicts the absolute pressure increment
                // directly (pSM = p_prev + dpML), so there is no trend to
                // extrapolate — the initial guess is simply the previous pressure.
                if (predictDP)
                {
                    pExtrap = p_prev;
                }
                else
                {
                    pExtrap = p_prev + dpPrev;
                }
                pSM = pExtrap;
                p = pExtrap;
                UStarPrev = U;
            }

            const auto uEqnWallStart = std::chrono::steady_clock::now();
            #include "UEqn.H"
            stepUEqnWall += std::chrono::duration<scalar>
            (
                std::chrono::steady_clock::now() - uEqnWallStart
            ).count();

            if (firstSMInputStage){
                // Storing the velocity increment and divergence fields
                // ONLY at the first PIMPLE iteration
                // (These must be consistent inputs for the ML model)

                UStar = U;
                dUStar = UStar - UStarPrev;
                ddUStar = dUStar - dUStarPrev;
                
                // Only fields required by the enabled python_module inputs are
                // computed; the rest stay zero.
                if (wantDivUStar)
                {
                    divUFirstPred = fvc::div(UStar);
                }

                if (wantLaplaceDpPrev)
                {
                    laplace_dpPrev = fvc::laplacian(dpPrev);
                }
                if (wantUDotGradDpPrev)
                {
                    uDotGradDpPrev = U & grad_dpPrev;
                }
                if (wantGradDpPrevMag)
                {
                    gradDpPrevMag = mag(grad_dpPrev);
                }

                // Must use the first momentum predictor matrix while UEqn is in scope.
                if (needRAU)
                {
                    rAU_ML = 1.0/UEqn.A();
                }
                if (needHbyA)
                {
                    HbyA_ML = constrainHbyA(rAU_ML*UEqn.H(), U, p);
                }
                if (needDivHbyA)
                {
                    divHbyA_ML = fvc::div(HbyA_ML);
                }

                if (wantRAUGradDpPrev)
                {
                    rAUGradDpPrev_ML = rAU_ML * grad_dpPrev;
                }
                if (wantDivRAUGradDpPrev)
                {
                    divRAUGradDpPrev_ML = fvc::laplacian(rAU_ML, dpPrev);
                }
                if (wantResidual)
                {
                    pressureEqResidualp_ML = divHbyA_ML - fvc::laplacian(rAU_ML, p);
                }

                if (needGradpPrev)
                {
                    grad_pPrev = fvc::grad(p_prev);
                }
                if (wantRAUGradpPrev)
                {
                    rAUGradpPrev_ML = rAU_ML * grad_pPrev;
                }
                if (wantDivRAUGradpPrev)
                {
                    divRAUGradpPrev_ML = fvc::laplacian(rAU_ML, p_prev);
                }

                if (wantDHbyA)
                {
                    dHbyA_ML = HbyA_ML - HbyA_ML_prev;
                }
                if (wantDDivHbyA)
                {
                    dDivHbyA_ML = divHbyA_ML - divHbyA_ML_prev;
                }

            }

            if (firstSMInputStage && surrogateActive)
            {
                struct timespec tw1, tw2;
                double posix_wall;
                clock_gettime(CLOCK_MONOTONIC, &tw1);
                surrogate.predict();
                clock_gettime(CLOCK_MONOTONIC, &tw2);
                posix_wall = 1000.0*tw2.tv_sec + 1e-6*tw2.tv_nsec
                           - (1000.0*tw1.tv_sec + 1e-6*tw1.tv_nsec);
                stepSMWall += posix_wall/1000.0;
                Info<< "DL pressure prediction: "
                    << posix_wall << " ms" << nl;

                if (smInputOnlyStage)
                {
                    U = U_time_prev;
                    phi = smInputPhiStart();
                    Info<< "[SM INPUT] restored timestep-start U/phi; "
                        << "frozen predictor inputs retained"
                                << " UStartError=" << gMax
                                    (mag(U.primitiveField() - U_time_prev.primitiveField()))
                                << " phiStartError=" << gMax
                                    (mag(phi.primitiveField() - smInputPhiStart().primitiveField()))
                        << nl;
                }
            
                pSM = p_prev + dpML;
                p = pSM;
                if (predictDUOutput || (smUseVelocityExtrapolation && smVelocityExtrapolationAlpha > 0 && smHaveVelocityHistory))
                {
                    volVectorField smAppliedVelocityIncrement
                    (
                        IOobject("smAppliedVelocityIncrement", runTime.timeName(), mesh,
                            IOobject::NO_READ, IOobject::NO_WRITE, false),
                        ddUStarML
                    );
                    if (smUseVelocityExtrapolation)
                    {
                        const scalar previousDeltaT = smPreviousCompletedDeltaT;
                        const scalar currentDeltaT = runTime.deltaTValue();
                        const bool validHistory = smHaveVelocityHistory && std::isfinite(previousDeltaT)
                            && std::isfinite(currentDeltaT) && previousDeltaT > VSMALL && currentDeltaT > 0;
                        const scalar factor = validHistory
                            ? smVelocityExtrapolationAlpha*currentDeltaT/previousDeltaT : 0.0;
                        smAppliedVelocityIncrement = factor*(dUStarPrev + dUCorrPrev);
                        Info<< "[SM VELOCITY EXTRAPOLATION] alpha=" << smVelocityExtrapolationAlpha
                            << " deltaT=" << currentDeltaT << " previousDeltaT=" << previousDeltaT
                            << " history=" << validHistory << " factor=" << factor << nl;
                    }
                    U = U_time_prev + smAppliedVelocityIncrement;
                    Info<< (predictDUOutput ? "[SM JOINT]" : "[SM VELOCITY EXTRAPOLATION]")
                        << " pre-constraint increment checks: UError="
                        << gMax(mag(U.primitiveField() - U_time_prev.primitiveField() - smAppliedVelocityIncrement.primitiveField()))
                        << " pError="
                        << gMax(mag(p.primitiveField() - p_prev.primitiveField() - dpML.primitiveField())) << nl;
                    U.correctBoundaryConditions();
                    fvOptions.correct(U);
                    jointDUSM = U - U_time_prev;
                    phi = fvc::flux(U);
                    MRF.makeRelative(phi);
                    if (p.needReference())
                    {
                        fvc::makeRelative(phi, U);
                        adjustPhi(phi, U, p);
                        fvc::makeAbsolute(phi, U);
                    }
                        Info<< (predictDUOutput ? "[SM JOINT] timestep-start U + predicted dU; max dU="
                            : "[SM VELOCITY EXTRAPOLATION] timestep-start U + extrapolated dU; max dU=")
                        << gMax(mag(jointDUSM.primitiveField())) << nl;
                }

                smDiagnosticsActiveThisStep = true;

                if (postSMInjectionReconstruction)
                {
                    const auto reconstructionWallStart =
                        std::chrono::steady_clock::now();
                    if (smPressureForceThenReconstruct)
                    {
                        MRF.correctBoundaryVelocity(U);
                        fvVectorMatrix forceUEqn
                        (
                            fvm::ddt(U) + fvm::div(phi, U) + MRF.DDt(U)
                          + turbulence->divDevSigma(U) == fvOptions(U)
                        );
                        if (smOracleCompatibleIntegration)
                        {
                            forceUEqn.relax();
                        }
                        else
                        {
                            forceUEqn.relax(smStateReconstructionEquationRelaxation);
                        }
                        fvOptions.constrain(forceUEqn);
                        const volScalarField forceRAU(1.0/forceUEqn.A());
                        const volVectorField forceHbyA(constrainHbyA(forceRAU*forceUEqn.H(), U, p));
                        surfaceScalarField forcePhi
                        (
                            "smForcePhi",
                            fvc::flux(forceHbyA) + MRF.zeroFilter
                            (
                                fvc::interpolate(forceRAU)*fvc::ddtCorr(U, phi)
                            )
                        );
                        MRF.makeRelative(forcePhi);
                        if (p.needReference())
                        {
                            fvc::makeRelative(forcePhi, U);
                            adjustPhi(forcePhi, U, p);
                            fvc::makeAbsolute(forcePhi, U);
                        }
                        constrainPressure(p, U, forcePhi, forceRAU, MRF);
                        p.correctBoundaryConditions();
                        pSM = p;
                        fvScalarMatrix forcePressureEqn
                        (
                            fvm::laplacian(forceRAU, p) == fvc::div(forcePhi)
                        );
                        forcePressureEqn.setReference(pRefCell, pRefValue);
                        phi = forcePhi - forcePressureEqn.flux();
                        U = forceHbyA - forceRAU*fvc::grad(p);
                        U.correctBoundaryConditions();
                        fvOptions.correct(U);
                        Info<< "[SM JOINT] no-solve HbyA pressure force before reconstruction" << nl;
                    }
                    // Mimic the realistic oracle mode:
                    // first UEqn -> inject SM pressure only -> fixed-pressure
                    // U/phi reconstruction -> normal PIMPLE continuation.
                    label reconstructionCorrectorsUsed = 0;
                    scalar reconstructionResidual = GREAT;
                    scalar aitkenFactor = 1.1;
                    bool havePreviousReconstructionDelta = false;

                    volVectorField previousReconstructionDelta
                    (
                        IOobject
                        (
                            "smPreviousReconstructionDelta",
                            runTime.timeName(), mesh,
                            IOobject::NO_READ, IOobject::NO_WRITE, false
                        ),
                        U - U
                    );

                    for
                    (
                        label reconstructionCorr = 0;
                        reconstructionCorr < smStateReconstructionCorrectors;
                        ++reconstructionCorr
                    )
                    {
                        const volVectorField reconstructionUBefore
                        (
                            IOobject
                            (
                                "smReconstructionUBefore",
                                runTime.timeName(), mesh,
                                IOobject::NO_READ, IOobject::NO_WRITE, false
                            ),
                            U
                        );
                        const surfaceScalarField reconstructionPhiBefore
                        (
                            IOobject
                            (
                                "smReconstructionPhiBefore",
                                runTime.timeName(), mesh,
                                IOobject::NO_READ, IOobject::NO_WRITE, false
                            ),
                            phi
                        );

                        MRF.correctBoundaryVelocity(U);
                        tmp<fvVectorMatrix> tUEqnReconstruct
                        (
                            fvm::ddt(U)
                          + fvm::div(phi, U)
                          + MRF.DDt(U)
                          + turbulence->divDevSigma(U)
                         == fvOptions(U)
                        );
                        fvVectorMatrix& UEqnReconstruct =
                            tUEqnReconstruct.ref();
                        if (smOracleCompatibleIntegration)
                        {
                            UEqnReconstruct.relax();
                        }
                        else
                        {
                            UEqnReconstruct.relax(smStateReconstructionEquationRelaxation);
                        }
                        fvOptions.constrain(UEqnReconstruct);

                        if (smOracleCompatibleIntegration && !smPressureForceThenReconstruct && reconstructionCorr == 0)
                        {
                            const volScalarField patchRAU(1.0/UEqnReconstruct.A());
                            const volVectorField patchHbyA(constrainHbyA(patchRAU*UEqnReconstruct.H(), U, p));
                            surfaceScalarField patchPhi
                            (
                                "smPatchPhi",
                                fvc::flux(patchHbyA) + MRF.zeroFilter
                                (
                                    fvc::interpolate(patchRAU)*fvc::ddtCorr(U, phi)
                                )
                            );
                            MRF.makeRelative(patchPhi);
                            if (p.needReference())
                            {
                                fvc::makeRelative(patchPhi, U);
                                adjustPhi(patchPhi, U, p);
                                fvc::makeAbsolute(patchPhi, U);
                            }
                            constrainPressure(p, U, patchPhi, patchRAU, MRF);
                            p.correctBoundaryConditions();
                            pSM = p;
                        }
                        const auto reconstructionPerf = solve
                        (
                            UEqnReconstruct == -fvc::grad(p)
                        );
                        fvOptions.correct(U);
                        reconstructionResidual = cmptMax
                        (
                            reconstructionPerf.initialResidual()
                        );
                        reconstructionCorrectorsUsed = reconstructionCorr + 1;

                        const volScalarField rAUReconstruct
                        (
                            1.0/UEqnReconstruct.A()
                        );
                        const volVectorField HbyAReconstruct
                        (
                            constrainHbyA
                            (
                                rAUReconstruct*UEqnReconstruct.H(), U, p
                            )
                        );
                        surfaceScalarField phiHbyAReconstruct
                        (
                            "smPhiHbyAReconstruct",
                            fvc::flux(HbyAReconstruct)
                          + MRF.zeroFilter
                            (
                                fvc::interpolate(rAUReconstruct)
                               *fvc::ddtCorr(U, phi)
                            )
                        );
                        MRF.makeRelative(phiHbyAReconstruct);
                        if (p.needReference())
                        {
                            fvc::makeRelative(phiHbyAReconstruct, U);
                            adjustPhi(phiHbyAReconstruct, U, p);
                            fvc::makeAbsolute(phiHbyAReconstruct, U);
                        }
                        constrainPressure
                        (
                            p, U, phiHbyAReconstruct, rAUReconstruct, MRF
                        );
                        p.correctBoundaryConditions();

                        fvScalarMatrix pEqnReconstruct
                        (
                            fvm::laplacian(rAUReconstruct, p)
                         == fvc::div(phiHbyAReconstruct)
                        );
                        pEqnReconstruct.setReference(pRefCell, pRefValue);
                        phi = phiHbyAReconstruct - pEqnReconstruct.flux();
                        U = HbyAReconstruct
                          - rAUReconstruct*fvc::grad(p);
                        U.correctBoundaryConditions();
                        fvOptions.correct(U);

                        volVectorField reconstructionDelta
                        (
                            IOobject
                            (
                                "smReconstructionDelta",
                                runTime.timeName(), mesh,
                                IOobject::NO_READ, IOobject::NO_WRITE, false
                            ),
                            U - reconstructionUBefore
                        );

                        if
                        (
                            smAitkenStateReconstruction
                         && havePreviousReconstructionDelta
                        )
                        {
                            const volVectorField deltaChange
                            (
                                IOobject
                                (
                                    "smReconstructionDeltaChange",
                                    runTime.timeName(), mesh,
                                    IOobject::NO_READ, IOobject::NO_WRITE, false
                                ),
                                reconstructionDelta
                              - previousReconstructionDelta
                            );
                            scalar numerator = 0.0;
                            scalar denominator = 0.0;
                            forAll(U, celli)
                            {
                                numerator += mesh.V()[celli]
                                   *(previousReconstructionDelta[celli]
                                   & deltaChange[celli]);
                                denominator += mesh.V()[celli]
                                   *magSqr(deltaChange[celli]);
                            }
                            reduce(numerator, sumOp<scalar>());
                            reduce(denominator, sumOp<scalar>());
                            if (denominator > VSMALL)
                            {
                                aitkenFactor = min
                                (
                                    smAitkenMaxFactor,
                                    max
                                    (
                                        smAitkenMinFactor,
                                       -aitkenFactor*numerator/denominator
                                    )
                                );
                            }
                            U = reconstructionUBefore
                              + aitkenFactor*reconstructionDelta;
                            phi = reconstructionPhiBefore
                              + aitkenFactor*(phi - reconstructionPhiBefore);
                            U.correctBoundaryConditions();
                            fvOptions.correct(U);
                        }

                        previousReconstructionDelta = reconstructionDelta;
                        havePreviousReconstructionDelta = true;
                        Info<< "[SM RECONSTRUCTION] corrector="
                            << reconstructionCorrectorsUsed
                            << " UResidual=" << reconstructionResidual
                            << " aitkenFactor=" << aitkenFactor << nl;

                        if
                        (
                            smAdaptiveStateReconstruction
                         && reconstructionCorrectorsUsed
                            >= smMinStateReconstructionCorrectors
                         && reconstructionResidual
                            < smStateReconstructionResidualTolerance
                        )
                        {
                            break;
                        }
                    }

                    Info<< "[SM RECONSTRUCTION] completed with "
                        << reconstructionCorrectorsUsed
                        << " fixed-pressure correctors finalUResidual="
                        << reconstructionResidual << nl;

                    // Exclude feature-generation and reconstruction solver
                    // records from PIMPLE residualControl. The configured U
                    // and p tolerances will be evaluated from normal post-SM
                    // outer correctors only.
                    clearSMStageResiduals<vector>(mesh);
                    clearSMStageResiduals<scalar>(mesh);

                    Info<< "[SM RECONSTRUCTION] residual history cleared; "
                        << "PIMPLE residualControl starts with "
                        << (smInputOnlyStage ? "normal outer corrector 1"
                            : "the next normal outer corrector") << nl;

                    if (smInputOnlyStage)
                    {
                        smInputOnlyStage = false;
                        smInputOnlyStageDone = true;
                    }

                    // Normal PIMPLE always resumes with a fresh UEqn; the
                    // feature-only mode has not consumed an outer iteration.
                                        stepReconstructionWall += std::chrono::duration<scalar>
                                        (
                                                std::chrono::steady_clock::now()
                                            - reconstructionWallStart
                                        ).count();
                    continue;
                }
                else if (applySMPressureConsistencyCorrection)
                {
                    // Preserve the existing pressure-force and optional
                    // predicted-U correction mode unchanged.
                    const volScalarField dpSM = pSM - pExtrap;
                    const volScalarField rAUcorr(1.0/UEqn.A());
                    U = U - rAUcorr*fvc::grad(dpSM);
                    if (applySMPredictedUCorrection)
                    {
                        U += ddUStarML;
                    }
                    U.correctBoundaryConditions();
                    fvOptions.correct(U);
                    phi = fvc::flux(U);
                    MRF.makeRelative(phi);
                    if (p.needReference())
                    {
                        fvc::makeRelative(phi, U);
                        adjustPhi(phi, U, p);
                        fvc::makeAbsolute(phi, U);
                    }
                }
            }

            // --- Pressure corrector loop
            scalar smLastPressureResidual = GREAT;
            while (pimple.correct())
            {
                const label firstPressureSolve = smResidualCount<scalar>(mesh, p.name());
                const auto pEqnWallStart = std::chrono::steady_clock::now();
                #include "pEqn.H"           
                stepPEqnWall += std::chrono::duration<scalar>
                (
                    std::chrono::steady_clock::now() - pEqnWallStart
                ).count();
                counter++;
                smLastPressureResidual = smFirstResidualSince<scalar>(mesh, p.name(), firstPressureSolve);
                if (smOracleCompatibleIntegration && smAdaptivePressureCorrectors
                    && counter >= smMinPressureCorrectors && smLastPressureResidual < smPressureCorrectorTolerance)
                {
                    break;
                }
            }

            bool smTurbulenceCorrected = false;
            if (pimple.turbCorr())
            {
                const auto turbulenceWallStart =
                    std::chrono::steady_clock::now();
                laminarTransport.correct();
                turbulence->correct();
                stepTurbulenceWall += std::chrono::duration<scalar>
                (
                    std::chrono::steady_clock::now() - turbulenceWallStart
                ).count();
                smTurbulenceCorrected = true;
            }
            if (smOracleCompatibleIntegration)
            {
                const scalar currentUResidual = smFirstResidualSince<vector>(mesh, U.name(), smFirstUSolve);
                if (currentUResidual < smOuterUResidualTolerance && smLastPressureResidual < smOuterPResidualTolerance)
                {
                    if (!smTurbulenceCorrected)
                    {
                        const auto turbulenceWallStart = std::chrono::steady_clock::now();
                        laminarTransport.correct();
                        turbulence->correct();
                        stepTurbulenceWall += std::chrono::duration<scalar>
                        (
                            std::chrono::steady_clock::now() - turbulenceWallStart
                        ).count();
                    }
                    Info<< "[SM ORACLE COMPAT] outer residual tolerance satisfied: U="
                        << currentUResidual << " p=" << smLastPressureResidual
                        << " UTol=" << smOuterUResidualTolerance << " pTol=" << smOuterPResidualTolerance << nl;
                    break;
                }
            }
        }
        const scalar stepPimpleWall = std::chrono::duration<scalar>
        (
            std::chrono::steady_clock::now() - pimpleWallStart
        ).count();

        // Calculate pressure deltas AFTER all PIMPLE iterations are complete
        // These are the cumulative CFD-computed values for the entire time step
        dp = p - p_prev;
        ddp = dp - dpPrev;

        if (predictDP)
        {
            smDeltaField = pSM - p_prev;
            truthDeltaField = dp;
        }
        else
        {
            smDeltaField = pSM - pExtrap;
            truthDeltaField = ddp;
            dpSMFieldPtr() = pSM - p_prev;
            dpTruthFieldPtr() = dp;
        }

        if (needGradDp)
        {
            grad_dp = fvc::grad(dp);
        }

        // End-of-timestep "[SM DIAGNOSTICS]" block: compares the frozen
        // pExtrap/pSM (captured during firstPimpleIter()) against the fully
        // converged pStar = p, plus explicit p_prev-only and
        // p_prev+dpPrev baselines. See smEndOfStepDiagnostics.H for details.
        #include "smEndOfStepDiagnostics.H"

        // Calculate pressure-correction velocity increments AFTER all PIMPLE iterations are complete
        // dUCorr  = U - UStar         (velocity gained during pressure solve)
        // ddUCorr = dUCorr - dUCorrPrev  (second difference — training target)
        dUCorr  = U - UStar;
        ddUCorr = dUCorr - dUCorrPrev;
        if (predictDUOutput || smUseVelocityExtrapolation)
        {
            jointDUTruth = U - U_time_prev;
        }
        smHaveVelocityHistory = true;
        smPreviousCompletedDeltaT = runTime.deltaTValue();

        // --- Sampling, writing, and ML training ---
        int retrainStatus = dataSampler.update();
        
        // Initialize surrogate on first timestep if train=False (no training, just inference)
        if (!train && !surrogateActive)
        {
            Info<< "Initializing DL Surrogate Model (inference mode, train=False)." << nl;
            surrogate.init();
            surrogateActive = true;
            applyPimpleSMOverrides();
        }
        else if (retrainStatus == 1 && !surrogateActive)
        {
            // First training completed: initialise surrogate (loads weights too)
            Info<< "Initializing DL Surrogate Model." << nl;
            surrogate.init();
            surrogateActive = true;
            applyPimpleSMOverrides();
        }
        else if (retrainStatus == 2 && surrogateActive)
        {
            // Incremental retrain completed: reload updated weights into memory
            surrogate.reload();
        }

        runTime.write();

        const scalar stepTotalWall = std::chrono::duration<scalar>
        (
            std::chrono::steady_clock::now() - stepWallStart
        ).count();
        const auto timingPrecision = Info().precision();
        Info().precision(9);
        Info<< "[STEP TIMING] time=" << runTime.timeName()
            << " UEqn=" << stepUEqnWall << " s"
            << " pEqns=" << stepPEqnWall << " s"
            << " kOmega=" << stepTurbulenceWall << " s"
            << " reconstruction=" << stepReconstructionWall << " s"
            << " sm=" << stepSMWall << " s"
            << " pimpleTotal=" << stepPimpleWall << " s"
            << " total=" << stepTotalWall << " s" << nl;
        Info().precision(timingPrecision);

        Info<< "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
            << "  ClockTime = " << runTime.elapsedClockTime() << " s"
            << nl << endl;
    }

    Info<< "End\n" << endl;

    return 0;
}

// ************************************************************************* //
