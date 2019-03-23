import numpy

class AirfoilElement:
    def __init__(self, pos, rot, chord, profile_coords=None):
        self.profile_coords = profile_coords
        self.pos = pos
        self.chord = chord
        self.rot = rot
    
    def transformed_coords(self):
        airfoil = numpy.copy(self.profile_coords)

        airfoil *= self.chord

        theta = numpy.radians(self.rot)
        c, s = numpy.cos(theta), numpy.sin(theta)
        R = numpy.array(((c,-s), (s, c))) 
        airfoil = numpy.dot(airfoil, R.T)
        
        T = numpy.tile(self.pos, (airfoil.shape[0], 1))
        airfoil += T

        return airfoil

class Airfoil:
    def __init__(self, elems):
        self.elems = elems

    def set_profile_coords(self, profile_coords):
        for elem in self.elems:
            elem.profile_coords = profile_coords
