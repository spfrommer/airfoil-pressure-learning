import numpy
import os

import pygmsh

import dirs
import io_utils

class MeshScheme:
    def __init__(self, acls, scl, bb, refine=3):
        """
        Acls => characteristic lengths for the airfoil elements
        Scl => characteristic length for free space
        bb => bounding box in 2d
        """
        self.acls = acls
        self.scl = scl
        self.bb = bb
        self.refine = refine

def append_zero_zs(mat):
    z_coords = numpy.zeros((mat.shape[0], 1))
    return numpy.append(mat, z_coords, axis = 1)

def add_physicals(scene_poly, vol, geom):
    geom.add_physical_surface(vol[2][1], label='outlet')
    geom.add_physical_surface([vol[2][0], vol[2][2]], label='walls')
    geom.add_physical_surface(vol[2][3], label='inlet')
    geom.add_physical_surface(vol[2][4:], label='airfoil')
    geom.add_physical_surface([vol[0], scene_poly.surface], label='frontAndBack')
    geom.add_physical_volume(vol[1], label='volume')

def run_gmesh(geom):
    f = open(dirs.out_path("tmp", "airfoil.geo"), "w")
    f.write(geom.get_code()) 
    f.close()

    base_cmd = "gmsh -3 {0} -optimize -o {1} -v 1"
    cmd = base_cmd.format(dirs.out_path("tmp", "airfoil.geo"),
                          dirs.out_path("tmp", "airfoil.msh")) 
    return io_utils.run_command(cmd.split(" "))

def mesh(airfoil, mesh_scheme):
    geom = pygmsh.built_in.Geometry()

    # Create polygons for airfoil elements
    def add_poly(elem, i):
        return geom.add_polygon(append_zero_zs(elem.transformed_coords()),
                                mesh_scheme.acls[i], make_surface=True) 
    airfoil_polys = [add_poly(e, i) for i,e in enumerate(airfoil.elems)]

    # Create bounding box
    boundbox = append_zero_zs(mesh_scheme.bb)
    
    # Create polygon for whole scene
    scene_poly = geom.add_polygon(boundbox, mesh_scheme.scl, holes=airfoil_polys)
    geom.add_raw_code("Recombine Surface {{{}}};".format(scene_poly.surface.id))
    
    # Extrude to 3D with one cell along z axis
    vol = geom.extrude(scene_poly.surface, translation_axis=[0, 0, 0.1],
                       num_layers=1, recombine=True)
    add_physicals(scene_poly, vol, geom)

    # This fixes a weird end of file in string error
    # That presumably comes from an encoding bug in gmsh
    geom.add_raw_code("\n")
    
    out = run_gmesh(geom)
    out["outPath"] = dirs.out_path("tmp", "airfoil.msh") 

    return out
