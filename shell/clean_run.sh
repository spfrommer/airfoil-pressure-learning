#!/bin/sh
# Removes all pyfoam/pygmsh output and temp files

cd case
# Delete all two-length numeric dicts
find . -regextype sed -regex ".*/[0-9][0-9]" -type d -exec rm -rf {} ';' 2>/dev/null 
# Delete all three-length numeric dicts
find . -regextype sed -regex ".*/[0-9][0-9][0-9]" -type d -exec rm -rf {} ';' 2>/dev/null 
# Delete all four-length numeric dicts
find . -regextype sed -regex ".*/[0-9][0-9][0-9][0-9]" -type d -exec rm -rf {} ';' 2>/dev/null 
# Delete all five-length numeric dicts
find . -regextype sed -regex ".*/[0-9][0-9][0-9][0-9][0-9]" -type d -exec rm -rf {} ';' 2>/dev/null 

rm -rf postProcessing
rm -rf constant/polyMesh
cd ..

rm -f out/tmp/*

rm -f src/*.pyc

rm -rf case/PyFoam*

rm -rf case/VTK
