from .galactic import Galactic as Galactic
from .supergalactic import Supergalactic as Supergalactic
from astropy.coordinates.baseframe import frame_transform_graph as frame_transform_graph
from astropy.coordinates.matrix_utilities import matrix_transpose as matrix_transpose, rotation_matrix as rotation_matrix
from astropy.coordinates.transformations import StaticMatrixTransform as StaticMatrixTransform

def gal_to_supergal(): ...
def supergal_to_gal(): ...
