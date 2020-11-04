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
        # eon energy = 3/2 * ne * kTe
        self.ergy_e = 1.5*KB_EV*np.multiply(pla.ne, pla.Te)
        
        
    def __str__(self):
        """Print eon energy module."""
        return f'energy flux = {self.qdfluxe}'
    
    def calc_th_cond_coeff(self, pla):
        """
        Calc thermal conduction coefficient.

        pla: Plasma2d object
        heat_cond_e: W/m/K, (nz, nx) matrix, heat conductivity for eon
        th_cond_e depend only on pla.
        """
        # calc thermal conductivity for eon
        self.th_cond_e = np.ones_like(pla.ne)*1e-3
    
    def calc_th_flux(self, pla, txp):
        """
        Calc eon thermal flux, Qe.
        
        Qe = 5/2kTe * fluxe - ke * dTe/dx
        dQe = 5/2kTe * dfluxe - ke * d2Te/dx2
        pla: Plasma2d object
        txp: Transp2d object
        """
        # calc convection term
        self.Qex = 2.5*KB_EV*np.multiply(self.Te, txp.fluxex)
        self.Qez = 2.5*KB_EV*np.multiply(self.Te, txp.fluxez)
        self.dQe = 2.5*KB_EV*np.multiply(self.Te, txp.dfluxe)
        # calc conduction term
        self.dTex, self.dTez = pla.geom.cnt_diff(self.Te)
        self.d2Te = pla.geom.cnt_diff_2nd(self.Te)
        self.Qex -= np.multiply(self.th_cond_e, self.dTex)
        self.Qez -= np.multiply(self.th_cond_e, self.dTez)
        self.dQe -= np.multiply(self.th_cond_e, self.d2Te)
        
    def calc_Te(self, delt, pla, pwr):
        """Calc Te."""
        self.ergy_e += (-self.dQe + pwr.input)*delt
        self.Te = np.divide(self.ergy_e, pla.ne)/1.5/KB_EV
        
    def plot_Te(self, pla):
        """
        Plot eon temperature.
        
        pla: Plasma_1d object
            use pla.geom.x for plot
        """
        x = pla.geom.x
        fig, axes = plt.subplots(1, 2, figsize=(8, 4),
                                 constrained_layout=True)
        # plot eon temperature
        ax = axes[0]
        ax.plot(x, self.Te, 'bo-')
        ax.legend(['e Temperature'])
        #
        ax = axes[1]
        ax.plot(x, self.Te, 'bo-')
        ax.legend(['e Temperature'])
        plt.show()
        
if __name__ == '__main__':
    """Test Eergy_1d."""
    from Mesh import Mesh_1d
    from Plasma1d import Plasma_1d
    from Transp1d import Ambi_1d
    from React1d import React_1d
    from Power1d import Power_1d
    mesh1d = Mesh_1d('Plasma_1d', 10e-2, nx=51)
    print(mesh1d)
    pla1d = Plasma_1d(mesh1d)
    pla1d.init_plasma()
    pla1d.plot_plasma()
    # calc the transport 
    txp1d = Ambi_1d(pla1d)
    txp1d.calc_transp_coeff(pla1d)
    txp1d.plot_transp_coeff(pla1d)
    # calc source term
    src1d = React_1d(pla1d)
    #
    ne_ave, ni_ave = [], []
    time = []
    dt = 1e-6
    niter = 3000
    for itn in range(niter):
        txp1d.calc_ambi(pla1d)
        pla1d.den_evolve(dt, txp1d, src1d)
        pla1d.bndy_plasma()
        pla1d.limit_plasma()
        # ne_ave.append(np.mean(pla1d.ne))
        # ni_ave.append(np.mean(pla1d.ni))
        # time.append(dt*(niter+1))
        # if not (itn+1) % (niter/10):
        #     txp1d.plot_flux(pla1d)
        #     pla1d.plot_plasma()
    # Test Eergy_1d
    dt = 3e-7
    niter = 100000
    een1d = Eergy_1d(pla1d)
    pwr1d = Power_1d(pla1d)
    pwr1d.calc_pwr_in(pla1d)
    een1d.plot_Te(pla1d)
    for itn in range(niter):
        een1d.calc_th_cond_coeff(pla1d)
        een1d.calc_th_flux(pla1d, txp1d)
        een1d.calc_Te(dt, pla1d, pwr1d)
        een1d.bndy_Te()
        if not (itn+1) % (niter/10):
            een1d.plot_Te(pla1d)
            