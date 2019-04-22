import os
import sys
import math
import glob
import numpy
import json
from lxml import etree
import cPickle as pickle
from subprocess import call

import dirs

# Adapted from Robert Lee's OpenFOAM tutorials
# https://github.com/openfoamtutorials/OpenFOAM_Tutorials_/blob/master/HowToPlotForces/plot_forces.py 
def read_forces(forces_file):
    def line2dict(line):
            tokens_unprocessed = line.split()
            tokens = [x.replace(")","").replace("(","")
                      for x in tokens_unprocessed]
            floats = [float(x) for x in tokens]
            data_dict = {}
            data_dict['time'] = floats[0]
            force_dict = {}
            force_dict['pressure'] = floats[1:4]
            force_dict['viscous'] = floats[4:7]
            force_dict['porous'] = floats[7:10]
            moment_dict = {}
            moment_dict['pressure'] = floats[10:13]
            moment_dict['viscous'] = floats[13:16]
            moment_dict['porous'] = floats[16:19]
            data_dict['force'] = force_dict
            data_dict['moment'] = moment_dict
            return data_dict

    time = []
    drag = []
    lift = []
    moment = []
    with open(forces_file,"r") as datafile:
        for line in datafile:
            if line[0] == "#":
                continue
            data_dict = line2dict(line)
            time += [data_dict['time']]
            drag += [data_dict['force']['pressure'][0] + data_dict['force']['viscous'][0]]
            lift += [data_dict['force']['pressure'][1] + data_dict['force']['viscous'][1]]
            moment += [data_dict['moment']['pressure'][2] + data_dict['moment']['viscous'][2]]
        datafile.close()

    return { "time": time, "drag": drag,
             "lift": lift, "moment": moment }

def write_airfoil_xml(airfoil, path):
    root = etree.Element("airfoil")
    elements = etree.Element("elements")
    root.append(elements)

    for elem in airfoil.elems:
        element = etree.Element("element")
        coordinates = etree.Element("coordinates")

        elements.append(element)
        element.append(coordinates)
        
        def append_coordinate(xy):
            point = etree.Element("point")
            x = etree.Element("x")
            y = etree.Element("y")

            coordinates.append(point)
            point.append(x)
            point.append(y)

            x.text = str(xy[0])
            y.text = str(xy[1])

        transformed = elem.transformed_coords()
        numpy.apply_along_axis(append_coordinate,
                               axis=1, arr=transformed)

    xml = etree.tostring(root, pretty_print=True) 
    with open(path, "w") as f:
        f.write(xml)

def write_dict_json(dict, path):
    with open(path, "w") as f:
        json.dump(dict, f)
        
def write_data(data, path):
    pickle.dump(data, open(path, "w"))

def read_data(path):
    return pickle.load(open(path, "r"))

# Runs a command from the root directory of the project
def run_command(argv):
    fatalError = False
    keyboardInterrupt = False
    try:
        call(argv, cwd=dirs.ROOT_DIR)
    except KeyboardInterrupt:
        keyboardInterrupt = True
    except Exception as e:
        fatalError = True

    return { "fatalError": fatalError,
             "keyboardInterrupt": keyboardInterrupt,
             "commandLine": " ".join(argv) }

def empty_dir(path):
    files = glob.glob(path + '/*')
    for f in files:
        os.remove(f)
