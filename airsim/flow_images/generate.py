import sys
import os
import os.path as op
path = op.dirname(op.dirname(op.dirname(op.abspath(__file__))))
print('Setting project root path: ' + path)
sys.path.append(path)

from colorama import init, Fore, Back, Style
from vtk import *
import numpy as np
import random

from airsim import sim
from airsim import presets
from airsim import dirs
from airsim import io_utils
from airsim.airfoil import Airfoil, AirfoilElement

import render

# Initialize terminal colors
init()

def load_coords(airfoil_name, truncate):
    coords = np.loadtxt(open(dirs.res_path(airfoil_name), "rb"), skiprows=1)
    coords = np.vstack((coords[:, 0], np.negative(coords[:, 1])))
    coords = np.transpose(coords)
    coords = np.flipud(coords)
    if truncate:
        coords = coords[2:-2,:]
    return coords

for airfoil_name in os.listdir(dirs.RES_DIR):
    analyzed = [a for a in os.listdir(dirs.out_path("images")) if
                  a.startswith(airfoil_name)]
    print("\n" + Fore.MAGENTA + "Analyzing: " + airfoil_name)
    print(Fore.MAGENTA + "Already analyzed: " + str(len(analyzed)))

    gen_num = 10 - len(analyzed)
    angles = np.round(np.random.uniform(-25, 25, size=(gen_num)),3).tolist()
    chords = np.round(np.random.uniform(0.3, 1, size=(gen_num)),3).tolist() 

    for angle, chord in zip(angles, chords):
        dir_name = "{}_{}_{}".format(airfoil_name, angle, chord) 
        dir_path = dirs.out_path("images", dir_name)
        if not op.isdir(dir_path):
            coords = load_coords(airfoil_name, False)
            airfoil = Airfoil([AirfoilElement([0,0], angle, chord)])
            airfoil.set_profile_coords(coords);

            try:
                simOut = sim.analyze_airfoil(airfoil,
                             presets.coarse_mesh_scheme(airfoil))
                os.mkdir(dir_path)

                render.render_sim(dir_path)
                io_utils.write_dict_json(simOut,
                                op.join(dir_path, "stats.txt"))
            except KeyboardInterrupt as e:
                print("\n" + Fore.RED + "Keyboard interrupt")
                sys.exit();
            except BaseException as e:
                print("\n" + Fore.RED +
                        "Failed on airfoil: " + airfoil_name)
        else:
            print("\n" + Fore.MAGENTA +
                  "Skipping, already analyzed: " + dir_name)
