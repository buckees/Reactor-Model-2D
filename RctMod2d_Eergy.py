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
import matplotlib.pyplot as plt
from copy import deepcopy


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
        self.pwr = deepcopy(pwr.input)
    
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

    def _set_nonPlasma(self, pla, Te_bc=0.1):
        """Impose fixed Te on the non-plasma materials."""
        for _idx, _mat in np.ndenumerate(pla.mesh.mat):
            if _mat:
                self.Te[_idx] = Te_bc
        
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
        self._set_nonPlasma(pla)
        
