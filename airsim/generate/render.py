from __future__ import print_function

from colorama import Fore, Back, Style
from PyFoam.Execution.UtilityRunner import UtilityRunner
from PyFoam.RunDictionary.SolutionDirectory import SolutionDirectory

import os.path as op
from ctypes import *
import numpy as np
from vtk import *

import airsim.dirs as dirs

# The openfoam case for pyfoam
case = SolutionDirectory(dirs.root_path("case"),
                         archive=None, paraviewLink=False)

def make_color_transfer(scalar_range, human_colored):
    color_transfer = vtkColorTransferFunction()
    if human_colored:
        color_transfer.AddRGBPoint(scalar_range[0], 0, 0, 1)
        color_transfer.AddRGBPoint(0, 0.5, 0.5, 0.5)
        color_transfer.AddRGBPoint(scalar_range[1], 1, 0, 0)
        color_transfer.SetBelowRangeColor(0, 0, 1)
        color_transfer.SetAboveRangeColor(1, 0, 0)
    else:
        delta = (scalar_range[1] - scalar_range[0]) / 256.0
        for i in range(256):
            range_interp = scalar_range[0] + delta * i
            color_transfer.AddRGBPoint(range_interp, 1, i / 256.0, 0)
            color_transfer.AddRGBPoint(range_interp + delta - 0.00000000001, 1, i / 256.0, 1)

        color_transfer.SetBelowRangeColor(0, 0, 1)
        color_transfer.SetAboveRangeColor(0, 1, 0)

    color_transfer.SetUseAboveRangeColor(True)
    color_transfer.SetUseBelowRangeColor(True)
    color_transfer.SetColorSpaceToRGB()
    return color_transfer

def render_sim(dir_path):
    print("\n" + Fore.BLUE + "Converting output to vtk...")
    res = UtilityRunner(argv=["foamToVTK", "-case", case.name],
                        silent=True).start()
    scalar_range = [-1000, 1000]
    color_transfer = make_color_transfer(scalar_range, False)
    run_vtk('p', 0, scalar_range, color_transfer, op.join(dir_path, 'p.png'))

    scalar_range = [-100, 100]
    color_transfer = make_color_transfer(scalar_range, False)
    run_vtk('U', 0, scalar_range, color_transfer, op.join(dir_path, 'Ux.png'))
    run_vtk('U', 1, scalar_range, color_transfer, op.join(dir_path, 'Uy.png'))

    
def run_vtk(field, component, scalar_range, color_transfer, png_path):
    case.reread()
    vtkFile = dirs.root_path("case", "VTK",
            "case_{}.vtk".format(case.last))

    reader = vtkUnstructuredGridReader()
    reader.SetFileName(vtkFile)
    reader.Update()

    cell_filter = vtkFieldDataToAttributeDataFilter()
    cell_filter.SetInputConnection(reader.GetOutputPort())
    cell_filter.SetInputFieldToCellDataField()
    cell_filter.SetOutputAttributeDataToCellData()
    cell_filter.SetScalarComponent(0, field, component)

    mapper = vtkDataSetMapper()
    mapper.SetLookupTable(color_transfer)
    mapper.ScalarVisibilityOn()
    mapper.SetInputConnection(cell_filter.GetOutputPort())

    actor = vtkActor()
    actor.SetMapper(mapper)

    camera = vtkCamera()
    camera.SetPosition(0, 0, 7)
    camera.SetFocalPoint(0, 0, 0)

    renderer = vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetActiveCamera(camera)

    window = vtkRenderWindow()
    window.SetMultiSamples(0)
    window.AddRenderer(renderer)
    window.SetWindowName("Figure")

    # Interactor is necessary to correctly save image
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(window)
    interactor.Initialize()

    window.Render()

    #data = window.GetRGBAPixelData(0, 0, 500, 500, 1)
    #converted = np.ctypeslib.as_array(data, shape=(500,500)) 
    #D = 500 * 500;
    #address = int(data[1:].split('_', 1)[0], 16)
    #print(type(data))
    #print(data)
    #print(data[1:].split('_', 1)[0])
    #print(c_float * D)
    #print((c_float * D).from_address(int(address)))
    #converted = np.frombuffer((c_float * D).from_address(address), np.float32).copy()
    #print(converted.shape)
    #print(converted)

    image_filter = vtkWindowToImageFilter()
    image_filter.SetInput(window)
    image_filter.SetScale(2,2)
    image_filter.SetInputBufferTypeToRGB()
    image_filter.ReadFrontBufferOff()
    image_filter.Update()

    writer = vtkPNGWriter()
    writer.SetFileName(png_path)
    writer.SetInputConnection(image_filter.GetOutputPort())
    writer.Write()
