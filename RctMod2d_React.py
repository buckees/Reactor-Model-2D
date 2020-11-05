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
from copy import copy, deepcopy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
colMap = copy(cm.get_cmap("jet"))
colMap.set_under(color='white')

class React2d(object):
    """Define the base tranport module/object."""
    
    def __init__(self, pla):
        """Import geometry information."""
        self.Se = np.zeros_like(pla.ne)  # initial eon flux
        self.Si = np.zeros_like(pla.ne)  # initial ion flux
    
    def calc_src(self, pla, ke=2.34e-14):
        """Calc src due to ionization."""
        self.Se = ke * np.power(pla.Te, 0.59)
        self.Se *= np.exp(-17.8/pla.Te)
        self.Se *= np.multiply(pla.ne, pla.nn)
        self.Si = deepcopy(self.Se)
        self._set_bc(pla)
        self._set_nonPlasma(pla)
        
    def _set_bc(self, pla):
        """Impose b.c. on the src."""
        for _idx in pla.mesh.bndy_list:
            self.Se[_idx] = 0.0
            self.Si[_idx] = 0.0

    def _set_nonPlasma(self, pla):
        """Impose fixed Te on the non-plasma materials."""
        for _idx, _mat in np.ndenumerate(pla.mesh.mat):
            if _mat:
                self.Se[_idx] = 0.0
                self.Si[_idx] = 0.0
        
    def plot_src(self, pla, figsize=(8, 8), ihoriz=1, 
                    dpi=300, fname='Power.png', imode='Contour'):
        """
        Plot power vs. position.
            
        var include input power and total power.
        figsize: a.u., (2, ) tuple, size of fig
        ihoriz: a.u., var, 0 or 1, set the layout of fig horizontal or not
        dpi: a.u., dots per inch
        fname: str, var, name of png file to save
        imode: str, var, ['Contour', 'Scatter']
        """
        _x, _z = pla.mesh.x, pla.mesh.z
        if ihoriz:
            fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi,
                                     constrained_layout=True)
        else:
            fig, axes = plt.subplots(2, 1, figsize=figsize, dpi=dpi,
                                     constrained_layout=True)
        
        # plot densities
        if imode == 'Contour':
            for _ax, _den, _title in zip(axes, (self.Se, self.Si), 
                                ('Eon Source', 'Ion Source')):
                _cs = _ax.contourf(_x, _z, _den, cmap=colMap)
                _ax.set_title(_title)
                fig.colorbar(_cs, ax=_ax, shrink=0.9)
            
        elif imode == 'Scatter':
            for _ax, _den, _title in zip(axes, (self.Se, self.Si), 
                                ('Eon Source', 'Ion Source')):
                _ax.scatter(_x, _z, c=_den, s=5, cmap=colMap)
                _ax.set_title(_title)
            
        for ax in axes:
            ax.set_xlabel('Position (m)')
            ax.set_ylabel('Height (m)')
            ax.set_aspect('equal')
        fig.savefig(fname, dpi=dpi)
        plt.close()