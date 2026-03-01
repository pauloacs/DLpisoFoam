#include "MLSampling.H"
#include "OFstream.H"
#include "POSIX.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

DataSampler::DataSampler
(
    const Foam::fvMesh& mesh,
    Foam::volVectorField& delta_U,
    Foam::volScalarField& delta_p_rgh,
    const std::string& dataDir,
    int warmUpSteps,
    int burstSteps,
    int regularInterval,
    int retrainInterval
)
:
    warmUpSteps_(warmUpSteps),
    burstSteps_(burstSteps),
    regularInterval_(regularInterval),
    retrainInterval_(retrainInterval),
    timeStep_(0),
    sampleCount_(0),
    samplesSinceRetrain_(0),
    initialTrainingDone_(false),
    mesh_(mesh),
    delta_U_(delta_U),
    delta_p_rgh_(delta_p_rgh),
    dataDir_(dataDir)
{}


bool DataSampler::shouldSample() const
{
    // Phase 1: warm-up — no sampling
    if (timeStep_ <= warmUpSteps_)
    {
        return false;
    }

    // Phase 2: burst sampling — every step for burstSteps_ steps
    int stepsSinceWarmUp = timeStep_ - warmUpSteps_;
    if (stepsSinceWarmUp <= burstSteps_)
    {
        return true;
    }

    // Phase 3: regular interval sampling
    int stepsSinceBurst = timeStep_ - (warmUpSteps_ + burstSteps_);
    return (stepsSinceBurst % regularInterval_ == 0);
}


bool DataSampler::shouldRetrain() const
{
    return (samplesSinceRetrain_ >= retrainInterval_);
}


void DataSampler::writeFieldData
(
    const Foam::volVectorField& vf,
    const Foam::volScalarField& sf,
    int step
)
{
    // Write vector field (delta_U) components and scalar field (delta_p_rgh)
    // as a simple CSV: x, y, z, Ux, Uy, Uz, p
    Foam::fileName outFile
    (
        dataDir_ + "/sample_" + Foam::name(step) + ".csv"
    );

    Foam::OFstream os(outFile);
    os << "x,y,z,Ux,Uy,Uz,p" << Foam::nl;

    const Foam::vectorField& cc = mesh_.C().internalField();
    const Foam::vectorField& Uvals = vf.internalField();
    const Foam::scalarField& pvals = sf.internalField();

    forAll(cc, cellI)
    {
        os  << cc[cellI].x() << ","
            << cc[cellI].y() << ","
            << cc[cellI].z() << ","
            << Uvals[cellI].x() << ","
            << Uvals[cellI].y() << ","
            << Uvals[cellI].z() << ","
            << pvals[cellI] << Foam::nl;
    }
}


void DataSampler::writeSample()
{
    // Ensure directory exists
    if (!Foam::isDir(dataDir_))
    {
        Foam::mkDir(dataDir_);
    }

    writeFieldData(delta_U_, delta_p_rgh_, timeStep_);

    sampleCount_++;
    samplesSinceRetrain_++;

    Foam::Info
        << "  [DataSampler] Sample #" << sampleCount_
        << " written at step " << timeStep_ << Foam::nl;
}


bool DataSampler::update()
{
    timeStep_++;
    bool activateSurrogate = false;

    if (shouldSample())
    {
        writeSample();
    }

    if (shouldRetrain())
    {
        if (!initialTrainingDone_)
        {
            // First training pass
            Foam::Info
                << "  [DataSampler] Running initial ML training ("
                << sampleCount_ << " samples)." << Foam::nl;

            std::string cmd = "python3 " + dataDir_ + "/train_init.py "
                            + dataDir_;
            (void)std::system(cmd.c_str());

            initialTrainingDone_ = true;
            activateSurrogate = true;
        }
        else
        {
            // Incremental update
            Foam::Info
                << "  [DataSampler] Running ML update ("
                << samplesSinceRetrain_ << " new samples)." << Foam::nl;

            std::string cmd = "python3 " + dataDir_ + "/train_update.py "
                            + dataDir_;
            (void)std::system(cmd.c_str());
        }

        samplesSinceRetrain_ = 0;
    }

    return activateSurrogate;
}