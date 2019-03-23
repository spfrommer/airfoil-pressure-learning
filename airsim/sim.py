from __future__ import print_function
import sys

from PyFoam.Execution.UtilityRunner import UtilityRunner
from PyFoam.Execution.BasicRunner import BasicRunner
from PyFoam.RunDictionary.SolutionDirectory import SolutionDirectory
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile

from colorama import Fore, Back, Style

import dirs
import mesh
import io_utils

# The openfoam case for pyfoam
case = SolutionDirectory(dirs.root_path("case"),
                         archive=None, paraviewLink=False)
# Matches the endTime in controlDict
max_iters = 2000

def print_res(res):
    print(Fore.WHITE + "Ran cmd: " + res["commandLine"])
    if res["keyboardInterrupt"]:
        raise KeyboardInterrupt
    if res["fatalError"]:
        print(Fore.RED + "Error in operation! " + 
                         "Please re-run command manually to debug.")
        raise RuntimeError("Failed operation")
    print(Fore.GREEN + "Success!")

def analyze_airfoil(airfoil, mesh_scheme):
    print("\n" + Fore.BLUE + "Cleaning run...")
    res = io_utils.run_command(["sh", dirs.shell_path("clean_run.sh"), ">/dev/null 2>&1"])
    print_res(res)

    # Generate mesh with gmsh and output to file
    print("\n" + Fore.BLUE + "Running gmsh...")
    res = mesh.mesh(airfoil, mesh_scheme)
    print_res(res)

    # Convert mesh to openfoam format
    print("\n" + Fore.BLUE + "Converting gmsh to openFoam...")
    res = UtilityRunner(argv=["gmshToFoam", dirs.out_path("tmp", "airfoil.msh"),
                              "-case", case.name], silent=True).start()
    print_res(res)

    # Defines type for boundaries
    print("\n" + Fore.BLUE + "Running changeDictionary on boundaries...")
    res = UtilityRunner(argv=["changeDictionary", "-case", case.name],
                        silent=True).start()
    print_res(res)

    # Run simulation
    print("\n" + Fore.BLUE + "Solving CFD with simpleFoam...")
    res = UtilityRunner(argv=["simpleFoam", "-case", case.name],
                        silent=True).start()
    print_res(res)
    print(Fore.WHITE + "Took {0} iterations and {1} secs"
                       .format(res["stepNr"], res["wallTime"]))
    if res["stepNr"] == max_iters:
        print(Fore.YELLOW + "Warning: openFoam did not converge")
        print("This is due to unsteady flow (likely too much turbulence)")
    
    # Read post-processing forces
    print("\n" + Fore.BLUE + "Reading force files...")
    forces = io_utils.read_forces(dirs.case_path("postProcessing",
                                  "airfoil_forces", "0", "forces.dat"))
    print(Fore.WHITE + "Downforce: {0}".format(forces["lift"][-1]))
    print("Drag: {0}".format(forces["drag"][-1]))

    return { "lift": forces["lift"][-1], "drag": forces["drag"][-1] }
