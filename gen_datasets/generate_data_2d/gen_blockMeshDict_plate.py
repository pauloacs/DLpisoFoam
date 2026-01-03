import argparse
import math
import matplotlib.pyplot as plt

def gen_blockMeshDict(x_cord, L, b, alpha, cell_scale, grading):
    """
    Create a `blockMeshDict` file for the geometry
    """
    grading = grading
    cell_scale = cell_scale

    scale = 1
    z = 0.05
    x_orig = 0
    y_orig = 0
    x_max = 0.75
    y_max = 0.05

    x_cord = x_cord
    alpha = math.pi*alpha/180
    L = L
    b = b

    #vertices of the rectangle

    if alpha > 0:
     A_cords = [x_cord - L/2*math.cos(alpha) + b*math.sin(alpha), y_orig + L/2*math.sin(alpha) + b*math.cos(alpha) ] #corresponds to point 3
     B_cords = [x_cord - L/2*math.cos(alpha) - b*math.sin(alpha), y_orig + L/2*math.sin(alpha) - b*math.cos(alpha) ] #corresponds to point 2
     C_cords = [x_cord + L/2*math.cos(alpha) + b*math.sin(alpha), y_orig - L/2*math.sin(alpha) + b*math.cos(alpha) ] #corresponds to point 10
     D_cords = [x_cord + L/2*math.cos(alpha) - b*math.sin(alpha), y_orig - L/2*math.sin(alpha) - b*math.cos(alpha) ] #corresponds to point 12
    else:
     alpha = -alpha    
     B_cords = [x_cord - L/2*math.cos(alpha) - b*math.sin(alpha), y_orig - L/2*math.sin(alpha) + b*math.cos(alpha) ] #corresponds to point 3
     D_cords = [x_cord - L/2*math.cos(alpha) + b*math.sin(alpha), y_orig - L/2*math.sin(alpha) - b*math.cos(alpha) ] #corresponds to point 2
     A_cords = [x_cord + L/2*math.cos(alpha) - b*math.sin(alpha), y_orig + L/2*math.sin(alpha) + b*math.cos(alpha)  ] #corresponds to point 10
     C_cords = [x_cord + L/2*math.cos(alpha) + b*math.sin(alpha), y_orig + L/2*math.sin(alpha) - b*math.cos(alpha)  ] #corresponds to point 12
     
    plt.scatter(A_cords[0], A_cords[1], label='A')
    plt.scatter(B_cords[0], B_cords[1], label='B')
    plt.scatter(C_cords[0], C_cords[1], label='C')
    plt.scatter(D_cords[0], D_cords[1], label='D')
    plt.legend()
    #plt.show()
    
    x_cell = int(A_cords[0] * cell_scale*200)
    x_cell2 = int((x_max - A_cords[0]) * cell_scale*200)
    y_cellAB = int((A_cords[1] - B_cords[1]) * cell_scale*200*3)
    y_cellBD = int((B_cords[1] - D_cords[1]) * cell_scale*200*3)
    y_cell2 = int((y_max - A_cords[1]) * cell_scale*200 *2)

    # Open file
    f = open("blockMeshDict", "w")

    # Write file
    f.write("/*--------------------------------*- C++ -*----------------------------------*\ \n"
            "| =========                |                                                  |\n"
            "| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox            |\n"
            "|  \\    /   O peration     | Version:  5                                      |\n"
            "|   \\  /    A nd           | Web:      www.OpenFOAM.org                       |\n"
            "|    \\/     M anipulation  |                                                  |\n"
            "\*---------------------------------------------------------------------------*/\n"
            "FoamFile\n"
            "{\n"
            "   version     2.0;\n"
            "   format      ascii;\n"
            "   class       dictionary;\n"
            "   object      blockMeshDict;\n"
            "}\n"
            "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n"
            "\n")
    f.write("convertToMeters {};\n".format(scale))
    f.write("\n"
            "vertices\n"
            "("
            "\n")
    f.write("    ({} {} {})\n".format(x_max, - y_max, z)) #0
    f.write("    ({} {} {})\n".format(x_orig, B_cords[1], z)) #1
    f.write("    ({} {} {})\n".format(B_cords[0], B_cords[1], z)) #2
    f.write("    ({} {} {})\n".format(A_cords[0], A_cords[1], z)) #3
    f.write("    ({} {} {})\n".format(x_orig, A_cords[1], z)) #4
    f.write("    ({} {} {})\n".format(x_orig, y_max, z)) #5
    f.write("    ({} {} {})\n".format(A_cords[0], y_max, z)) #6 
    f.write("    ({} {} {})\n".format(x_max, y_max, z)) #7
    f.write("    ({} {} {})\n".format(x_max, A_cords[1] , z)) #8
    f.write("    ({} {} {})\n".format(x_max, C_cords[1], z)) #9
    f.write("    ({} {} {})\n".format(C_cords[0], C_cords[1], z)) #10
    f.write("    ({} {} {})\n".format(x_orig, D_cords[1], z)) #11
    f.write("    ({} {} {})\n".format(D_cords[0], D_cords[1], z)) #12
    f.write("    ({} {} {})\n".format(x_orig, - y_max, z)) #13
    f.write("    ({} {} {})\n".format(D_cords[0], - y_max, z)) #14
    f.write("    ({} {} {})\n".format(x_max, D_cords[1], z)) #15


    f.write("    ({} {} {})\n".format(x_max, - y_max, -z)) #16
    f.write("    ({} {} {})\n".format(x_orig, B_cords[1], -z)) #17
    f.write("    ({} {} {})\n".format(B_cords[0], B_cords[1], -z)) #18
    f.write("    ({} {} {})\n".format(A_cords[0], A_cords[1], -z)) #19
    f.write("    ({} {} {})\n".format(x_orig, A_cords[1], -z)) #20
    f.write("    ({} {} {})\n".format(x_orig, y_max, -z)) #21
    f.write("    ({} {} {})\n".format(A_cords[0], y_max, -z)) #22
    f.write("    ({} {} {})\n".format(x_max, y_max, -z)) #23
    f.write("    ({} {} {})\n".format(x_max, A_cords[1] , -z)) #24
    f.write("    ({} {} {})\n".format(x_max, C_cords[1], -z)) #25
    f.write("    ({} {} {})\n".format(C_cords[0], C_cords[1], -z)) #26
    f.write("    ({} {} {})\n".format(x_orig, D_cords[1], -z)) #27
    f.write("    ({} {} {})\n".format(D_cords[0], D_cords[1], -z)) #28
    f.write("    ({} {} {})\n".format(x_orig, - y_max, -z)) #29
    f.write("    ({} {} {})\n".format(D_cords[0],  - y_max, -z)) #30
    f.write("    ({} {} {})\n".format(x_max, D_cords[1], -z)) #31


    f.write(");\n"
            "\n"
            "blocks\n"
            "(\n")
    f.write("    hex (17 18 19 20 1 2 3 4) ({} {} {}) simpleGrading ({} 1 1)\n".format(x_cell, y_cellAB, 1, 1/grading)) #block 0
    f.write("    hex (20 19 22 21 4 3 6 5) ({} {} {}) simpleGrading ({} ( (0.5 0.5 {}) (0.5 0.5 {}) ) 1)\n".format(x_cell, y_cell2 , 1, 1/grading, grading, 1/grading))

    f.write("    hex (19 24 23 22 3 8 7 6) ({} {} {}) simpleGrading ({} ( (0.5 0.5 {}) (0.5 0.5 {}) ) 1)\n".format(x_cell2, y_cell2, 1, grading, grading, 1/grading))
    f.write("    hex (26 25 24 19 10 9 8 3) ({} {} {}) simpleGrading ({} 1 1)\n".format( x_cell2 , y_cellBD, 1, grading)) #3

    f.write("    hex (27 28 18 17 11 12 2 1) ({} {} {}) simpleGrading ({} 1 1)\n".format(x_cell , y_cellBD, 1, 1/grading)) #4
    f.write("    hex (29 30 28 27 13 14 12 11) ({} {} {}) simpleGrading ({} ( (0.5 0.5 {}) (0.5 0.5 {}) ) 1)\n".format( x_cell , y_cell2, 1, 1/grading, grading, 1/grading)) #5

    f.write("    hex  (30 16 31 28 14 0 15 12) ({} {} {}) simpleGrading ({} ( (0.5 0.5 {}) (0.5 0.5 {}) ) 1)\n".format(x_cell2 , y_cell2, 1, grading, grading, 1/grading))
    f.write("    hex (28 31 25 26 12 15 9 10) ({} {} {}) simpleGrading ({} 1 1)\n".format( x_cell2 , y_cellAB, 1, grading))


    f.write(");\n"
            "\n"
            "edges\n"
            "(\n"
            ");\n"
            "\n"
            "boundary\n"
            "(\n"
            "    inlet\n"
            "    {\n"
            "        type patch;\n"
            "        faces\n"
            "        (\n"
            "            (20 4 5 21)\n"
            "            (17 1 4 20)\n"
            "            (27 11 1 17)\n"
            "            (29 13 11 27)\n"
            "        );\n"
            "    }\n"
            "    outlet\n"
            "    {\n"
            "        type patch;\n"
            "        faces\n"
            "        (\n"
            "            (7 8 24 23)\n"
            "            (8 9 25 24)\n"
            "            (9 15 31 25)\n"
            "            (15 0 16 31)\n"
            "        );\n"
            "    }\n"
            "    top\n"
            "    {\n"
            "        type wall;\n"
            "        faces\n"
            "        (\n"
            "            (21 5 6 22)\n"
            "            (22 6 7 23)\n"
            "            (13 29 30 14)\n"
            "            (14 30 16 0)\n"
            "        );\n"
            "    }\n"
            "    obstacle\n"
            "    {\n"
            "        type wall;\n"
            "        faces\n"
            "        (\n"
            "            (3 19 26 10)\n"
            "            (10 26 28 12)\n"
            "            (18 2 12 28)\n"
            "            (19 3 2 18)\n"
            "        );\n"
            "    }\n"
            ");\n"
            "\n"
            "mergePatchPairs\n"
            "(\n"
            ");\n"
            "\n"
            "// ************************************************************************* //\n")

    # Close file
    f.close()


# Total cell 7500

parser = argparse.ArgumentParser(description="Generating blockMeshDict file for the geometry")
parser.add_argument("x_cord", help="X coordinate of forward step")
parser.add_argument("L", help="length")
parser.add_argument("b", help="")
parser.add_argument("alpha", help="angle of attack")
parser.add_argument("cell_scale", help="define the refinement level")
parser.add_argument("grading", help="grading")


args = parser.parse_args()
gen_blockMeshDict(float(args.x_cord), float(args.L) , float(args.b), float(args.alpha), float(args.cell_scale), float(args.grading) )
