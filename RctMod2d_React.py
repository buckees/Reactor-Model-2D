"""
2D Plasma Transport Module.

Transp_1d contains:
    Diffusion only
    Ambipolar
    Drfit-Diffusion
    Momentum Solver

    Continuity Eq. dn/dt = -dF/dx * dt + S * det
    Input: depends on transport mode
    Output: dF/dx for continuity equation.
"""

import numpy as np

class React2d(object):
    """Define the base tranport module/object."""
    
    def __init__(self, pla):
        """Import geometry information."""
        self.se = np.zeros_like(pla.ne)  # initial eon flux
        self.si = np.zeros_like(pla.ne)  # initial ion flux
    
    def calc_src(self, pla, ke=1.0):
        """Calc src due to ionization."""
        self.ke = ke * np.sqrt(pla.Te)
        self.se = np.multiply(pla.ne, pla.nn)
        self.se = np.multiply(self.se, self.ke)
        self.si = np.multiply(pla.ne, pla.nn)
        self.si = np.multiply(self.si, self.ke)