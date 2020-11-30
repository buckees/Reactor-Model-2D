"""
2D Plasma Electron Energy Module.

Eergy_2d contains:
    Electron energy equation
    d(3/2nekTe)/dt = -dQ/dx + Power_in(ext.) - Power_loss(react)
    Input: ne, Te from Plasma1d, E_ext from field solver
    Output: Te
"""

from Constants import KB_EV, EON_MASS, UNIT_CHARGE

import numpy as np
from copy import copy, deepcopy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
colMap = copy(cm.get_cmap("jet"))
colMap.set_under(color='white')



class Eergy2d(object):
    """Define the eon energy module/object."""
    
    def __init__(self, pla):
        """Import Plasma1d information."""
        self.Te = deepcopy(pla.Te)
        self.pwr = np.zeros_like(pla.Te)
        # eon energy = 3/2 * ne * kTe
        self.ergy_e = 1.5*KB_EV*np.multiply(pla.ne, pla.Te)
        
        
    def __str__(self):
        """Print eon energy module."""
        return f'energy flux = {self.qdfluxe}'
    
    def get_pwr(self, pwr):
        """
        Get input power from Power2d().
        
        pwr: Power2d() object.
        """
        self.pwr = pwr.input
    
    def _calc_th_cond_coeff(self, pla):
        """
        Calc thermal conduction coefficient.

        pla: Plasma2d() object.
        heat_cond_e: W/m/K, (nz, nx) matrix, heat conductivity for eon
        th_cond_e depend only on pla.
        """
        # calc thermal conductivity for eon
        self.th_cond_e = np.ones_like(pla.ne)*1e-3
    
    def _calc_th_flux(self, pla, txp):
        """
        Calc eon thermal flux, Qe.
        
        Qe = 5/2kTe * fluxe - ke * dTe/dx
        dQe = 5/2kTe * dfluxe - ke * d2Te/dx2
        pla: Plasma2d() object
        txp: Transp2d() object
        """
        # calc convection term
        self.Qex = 2.5*KB_EV*np.multiply(self.Te, txp.fluxex)
        self.Qez = 2.5*KB_EV*np.multiply(self.Te, txp.fluxez)
        self.dQe = 2.5*KB_EV*np.multiply(self.Te, txp.dfluxe)
        # calc conduction term
        self.dTex, self.dTez = pla.mesh.cnt_diff(self.Te)
        self.d2Te = pla.mesh.cnt_diff_2nd(self.Te)
        self.Qex -= np.multiply(self.th_cond_e, self.dTex)
        self.Qez -= np.multiply(self.th_cond_e, self.dTez)
        self.dQe -= np.multiply(self.th_cond_e, self.d2Te)

    def _set_bc(self, pla, Te_bc=0.1):
        """Impose b.c. on the Te."""
        for _idx in pla.mesh.bndy_list:
            self.Te[_idx] = Te_bc

    def _set_nonPlasma(self, pla, Te_bc=0.1):
        """Impose fixed Te on the non-plasma materials."""
        for _idx, _mat in np.ndenumerate(pla.mesh.mat):
            if _mat:
                self.Te[_idx] = Te_bc

    def _limit_Te(self, T_min=0.001, T_max=100.0):
        """Limit Te in the plasma."""
        self.Te = np.clip(self.Te, T_min, T_max)
        
    def calc_Te(self, delt, pla, txp):
        """
        Calc Te.
        
        delt: s, var, time step for explict method
        pla: Plasma2d() object.
        txp: Transp2d() object.
        """
        self._calc_th_cond_coeff(pla)
        self._calc_th_flux(pla, txp)
        self.ergy_e += (-self.dQe + self.pwr)*delt
        self.Te = np.divide(self.ergy_e, pla.ne)/1.5/KB_EV
        self._set_bc(pla)
        self._set_nonPlasma(pla)
        self._limit_Te()

    def plot_dQe(self, pla, figsize=(8, 8), ihoriz=1, 
                    dpi=300, fname='dQe.png', imode='Contour'):
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
            for _ax, _den, _title in zip(axes, (self.dQe, self.pwr), 
                                ('Power Loss', 'Power to Eon')):
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
