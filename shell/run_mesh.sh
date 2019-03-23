# Assuming that you already ran gmsh and have an v2 airfoil.msh in the out directory
gmshToFoam out/tmp/airfoil.msh -case case
checkMesh -case case
changeDictionary -case case
simpleFoam -case case
