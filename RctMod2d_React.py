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

class React_2d(object):
    """Define the base tranport module/object."""
    
    def __init__(self, pla):
        """Import geometry information."""
        _x = pla.mesh.x
        self.se = np.zeros_like(_x)  # initial eon flux
        self.si = np.zeros_like(_x)  # initial ion flux
        