"""
2D Plasma Power Module.

Power2d contains:
    Power calculation
    Input: ne, Te from Plasma1d, E_ext from field solver
    Output: power
"""


import numpy as np
from copy import copy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
colMap = copy(cm.get_cmap("jet"))
colMap.set_under(color='white')

class Power2d(object):
    """Define the power module/object."""
    
    def __init__(self, pla):
        """Import Plasma1d information."""
        self.input = np.zeros_like(pla.ne)  # initial eon flux
        
        
    def __str__(self):
        """Print eon energy module."""
        return f'power input = {self.input}'
    
    def calc_pwr_in(self, pla, pwr=1.0, imode='Uniform'):
        """
        Calc input power.

        pla: Plasma2d object
             calc uses pla.Te,i and pla.coll_em
        pwr: W, var, total input power from external
        imode: str, var, ['Uniform', 'ne'], how input power is distributed 
              into the plasma
        input: W, (nz, nx) matrix, power input
        """
        if imode in ['Uniform', 'ne', 'EF', 'Top']:
            pass
        else:
            return print('imode is not recognized in calc_pwr_in()')
        pwr = pwr/pla.mesh.area
        if imode == 'Uniform':
            self.input = np.ones_like(pla.ne)*pwr
        elif imode == 'ne':
            self.input = pla.ne/pla.ne.sum()*pwr
        elif imode == 'EF':
            temp_EF = pla.ne*pla.EF*pla.EF
            self.input = temp_EF/temp_EF.sum()*pwr
        elif imode == 'Top':
            self.input = pla.mesh.z/pla.mesh.z.sum()*pwr
    
    def plot_pwr(self, pla, figsize=(8, 8), ihoriz=1, 
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
            for _ax, _den, _title in zip(axes, (self.input, self.input), 
                                ('Power input to E', 'Total power')):
                _cs = _ax.contourf(_x, _z, _den, cmap=colMap)
                _ax.set_title(_title)
                fig.colorbar(_cs, ax=_ax, shrink=0.9)
            
        elif imode == 'Scatter':
            for _ax, _den, _title in zip(axes, (self.Te, self.Ti), 
                                ('E Temperature', 'Ion Temperature')):
                _ax.scatter(_x, _z, c=_den, s=5, cmap=colMap)
                _ax.set_title(_title)
            
        for ax in axes:
            ax.set_xlabel('Position (m)')
            ax.set_ylabel('Height (m)')
            ax.set_aspect('equal')
        fig.savefig(fname, dpi=dpi)
        plt.close()