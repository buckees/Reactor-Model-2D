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

class Power_1d(object):
    """Define the eon energy module/object."""
    
    def __init__(self, pla):
        """Import Plasma1d information."""
        nx = pla.geom.nx
        self.input = np.zeros(nx)  # initial eon flux
        
        
    def __str__(self):
        """Print eon energy module."""
        return f'label = {self.qdfluxe}'
    
    def calc_pwr_in(self, pla):
        """
        Calc thermal conduction coefficient.

        pla: Plasma_1d object
             calc uses pla.Te,i and pla.coll_em
        heat_cond_e: W/m/K, heat conductivity for eon
        """
        # calc thermal conductivity for eon
        self.input = np.ones_like(pla.ne)*1.0