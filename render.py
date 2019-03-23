import os
from vtk import *

#set the fileName for the current case
myFileName = 'case/VTK/case_474.vtk'

#Need a reader for unstructured grids
reader = vtkUnstructuredGridReader()
reader.SetFileName(myFileName)
reader.Update()
#In OpenFOAM all results are Field-data.
#This has no concept of cells or nodes.
#Need to filter to cells.
toCellFilter = vtkFieldDataToAttributeDataFilter()
output = reader.GetOutput();
#toCellFilter.SetInput(reader.GetOutput())
toCellFilter.SetInputConnection(reader.GetOutputPort())
#toCellFilter.SetInputData(reader.GetOutput())
toCellFilter.SetInputFieldToCellDataField()
toCellFilter.SetOutputAttributeDataToCellData()

#Assign here which field
#we are interested in.
toCellFilter.SetScalarComponent(0,'U',1)
#This is all we need to do do calculations.
#To get 3D image, need some more components.
#First a window
renWin = vtkRenderWindow()

camera = vtkCamera();
camera.SetPosition(0, 0, 2);
camera.SetFocalPoint(0, 0, 0);

#Then a renderer. Renders data to an image.
ren1 = vtkRenderer()
ren1.SetActiveCamera(camera)
#Add renderer to window
renWin.AddRenderer(ren1)
renWin.SetWindowName("Figure")
#Add pressure data to the renderer.
#Mapping assigns data to colors and geometry.
mapper = vtkDataSetMapper()

minVal = -50;
maxVal = 50
mapper.SetScalarRange([minVal,maxVal])

colorTransfer = vtkColorTransferFunction()
colorTransfer.SetBelowRangeColor(1, 0, 0)
colorTransfer.SetAboveRangeColor(1, 1, 1)
colorTransfer.SetUseAboveRangeColor(True)
colorTransfer.SetUseBelowRangeColor(True)

#colorTransfer.AddRGBPoint(minVal, 1, 0, 0)
#colorTransfer.AddRGBPoint(0, 0.5, 0.5, 0.5)
#colorTransfer.AddRGBPoint(maxVal, 0, 0, 1)

#colorTransfer.AddRGBPoint(minVal, 1, 0, 0)
#colorTransfer.AddRGBPoint(maxVal, 1, 1, 1)

colorTransfer.AddRGBPoint(minVal, 0, 0, 0)
colorTransfer.AddRGBPoint(maxVal, 1, 1, 1)

colorTransfer.SetColorSpaceToRGB()
mapper.SetLookupTable(colorTransfer)
mapper.ScalarVisibilityOn()

#mapper.SetInput(toCellFilter.GetOutput())
mapper.SetInputConnection(toCellFilter.GetOutputPort())
#mapper.SetInputData(toCellFilter.GetOutput())

iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)

iren.Initialize()

#The object is assigned to an actor.
actor = vtkActor()
actor.SetMapper(mapper)
#Add actor to renderer.
ren1.AddActor(actor)

#Finally render image
renWin.Render()
#iren.Start()

windowToImageFilter = vtkWindowToImageFilter()
windowToImageFilter.SetInput(renWin)
windowToImageFilter.SetInputBufferTypeToRGB()
windowToImageFilter.ReadFrontBufferOff()
windowToImageFilter.Update()

writer = vtkPNGWriter()
writer.SetFileName('test.png')
writer.SetInputConnection(windowToImageFilter.GetOutputPort())
writer.Write()

