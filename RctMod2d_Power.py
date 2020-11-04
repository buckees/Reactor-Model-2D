"""
2D Plasma Power Module.

Power2d contains:
    Power calculation
    Input: ne, Te from Plasma1d, E_ext from field solver
    Output: power
"""


import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

class Power2d(object):
    """Define the power module/object."""
    
    def __init__(self, pla):
        """Import Plasma1d information."""
        _nx = pla.mesh.nx
        self.input = np.zeros(_nx)  # initial eon flux
        
        
    def __str__(self):
        """Print eon energy module."""
        return f'power input = {self.input}'
    
    def calc_pwr_in(self, pla):
        """
        Calc input power.

        pla: Plasma2d object
             calc uses pla.Te,i and pla.coll_em
        input: W, (nz, nx) matrix, power input
        """
        # calc thermal conductivity for eon
        self.input = np.ones_like(pla.ne)*1.0