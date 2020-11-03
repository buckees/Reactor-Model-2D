"""
2D Plasma Transport Module.

Transp_2d contains:
    Diffusion only
    Ambipolar
    Drfit-Diffusion
    Momentum Solver

    Continuity Eq. dn/dt = -dF/dx * dt + S * det
    Input: depends on transport mode
    Output: dF/dx for continuity equation.
"""

from Constants import KB_EV, EON_MASS, UNIT_CHARGE

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


class Transp_2d(object):
    """Define the base tranport module/object."""
    
    def __init__(self, pla):
        """Import geometry information."""
        _x = pla.geom.x
        self.fluxe = np.zeros_like(_x)  # initial eon flux
        self.fluxi = np.zeros_like(_x)  # initial ion flux
        self.dfluxe = np.zeros_like(_x)  # initial eon flux
        self.dfluxi = np.zeros_like(_x)  # initial ion flux

    def calc_transp_coeff(self, pla):
        """
        Calc diffusion coefficient and mobility.

        pla: Plasma_1d object
             calc uses pla.Te,i and pla.coll_em
        De,i: m^2/s, D = k*T/(m*coll_m)
        Mue,i: m^2/(V*s), Mu = q/(m*coll_m)
        """
        # calc diff coeff: D = k*T/(m*coll_m)
        self.De = np.divide(KB_EV*pla.Te, EON_MASS*pla.coll_em)  
        self.Di = np.divide(KB_EV*pla.Ti, pla.Mi*pla.coll_im)  
        # calc mobility: Mu = q/(m*coll_m)
        self.Mue = UNIT_CHARGE/EON_MASS/pla.coll_em
        self.Mui = UNIT_CHARGE/pla.Mi/pla.coll_im

    def plot_transp_coeff(self, pla, 
                          figsize=(8, 8), dpi=600, fname='Transp.png'):
        """
        Plot transp coeff.
        
        pla: Plasma_1d object
            use pla.geom.x for plot
        """
        _x, _z = self.geom.x, self.geom.z
        fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi,
                                 constrained_layout=True)
        # plot densities
        ax = axes[0]
        ax.scatter(_x, _z, c=self.De, s=1, cmap=colMap, vmin=0.2)
        ax.set_title('E Diffusion Coeff')
        ax.set_xlabel('Position (m)')
        ax.set_ylabel('Height (m)')
        ax = axes[1]
        ax.scatter(_x, _z, c=self.Di, s=1, cmap=colMap, vmin=0.2)
        ax.set_title('Ion Diffusion Coeff')
        ax.set_xlabel('Position (m)')
        ax.set_ylabel('Height (m)')
        fig.savefig(fname, dpi=dpi)

    
    def plot_flux(self, pla):
        """
        Plot flux and dflux.
        
        pla: Plasma_1d object
            use pla.geom.x for plot
        """
        x = pla.geom.x
        fig, axes = plt.subplots(1, 2, figsize=(8, 4),
                                 constrained_layout=True)
        # plot potential
        ax = axes[0]
        ax.plot(x, self.fluxe, 'bo-')
        ax.plot(x, self.fluxi, 'ro-')
        ax.legend(['e flux', 'Ion flux'])
        # plot E-field
        ax = axes[1]
        ax.plot(x, self.dfluxe, 'bo-')
        ax.plot(x, self.dfluxi, 'ro-')
        ax.legend(['e dflux', 'Ion dflux'])
        # show fig
        plt.show(fig)

class Diff_1d(Transp_1d):
    """
    Calc the dflux for Diffusion Only Module.
    
    dn/dt = -D * d2n/dx2 + Se
    D: m^2/s, diffusion coefficient is calc from Tranps_1d
    Output: D * d2n/dx2
    """
    def calc_diff(self, pla):
        """Calc diffusion term: D * d2n/dx2 and diffusion flux D * dn/dx. """
        # Calc transp coeff first
        self.calc_transp_coeff(pla)
        # Calc flux
        self.fluxe = -self.De * pla.geom.cnt_diff(pla.ne)
        self.fluxi = -self.Di * pla.geom.cnt_diff(pla.ni)
        # Calc dflux
        self.dfluxe = -self.De * pla.geom.cnt_diff_2nd(pla.ne)
        self.dfluxi = -self.Di * pla.geom.cnt_diff_2nd(pla.ni)
    
class Ambi_1d(Transp_1d):
    """
    Calc the dflux for Ambipolar Diffusion Module.

    dn/dt = -Da * d2n/dx2 + Se
    Di: m^2/s, ion diffusion coefficient is calc from Tranps_1d
    Da = Di(1 + Te/Ti).
    Output: Da * d2n/dx2 and E-field
    """

    def calc_ambi(self, pla):
        """
        Calc ambipolar diffusion coefficient.

        The ambipolar diffusion assumptions:
            1. steady state, dne/dt = 0. it cannot be used to
            describe plasma decay.
            2. ni is calculated from continuity equation.
            3. plasma is charge neutral, ne = ni
            4. Ionization Se is needed to balance diffusion loss.
        Da = (De*Mui + Di*Mue)/(Mue + Mui)
        Da = Di(1 + Te/Ti).
        Ea = (Di - De)/(Mui + Mue)*dn/dx/n
        Orginal Ambipolar Coeff Da = (De*Mui + Di*Mue)/(Mue + Mui)
        self.Da = (Plasma_1d.De*Plasma_1d.Mui + Plasma_1d.Di*Plasma_1d.Mue) / \
                  (Plasma_1d.Mue + Plasma_1d.Mui)
        Assume Te >> Ti, Ambipolar Coeff can be simplified as
        Da = Di(1 + Te/Ti).
        """
        # Calc transp coeff first
        self.calc_transp_coeff(pla)
        # Calc ambi coeff
        self.Da = self.Di*(1.0 + np.divide(pla.Te, pla.Ti))
        dni = pla.geom.cnt_diff(pla.ni)
        self.Ea = np.divide(self.Di - self.De, self.Mui + self.Mue)
        self.Ea *= np.divide(dni, pla.ni)
        # Calc flux
        self.fluxe = -self.Da * pla.geom.cnt_diff(pla.ne)
        self.fluxi = -self.Da * pla.geom.cnt_diff(pla.ni)
        # Calc dflux
        self.dfluxe = -self.Da * pla.geom.cnt_diff_2nd(pla.ne)
        self.dfluxi = -self.Da * pla.geom.cnt_diff_2nd(pla.ni)
        # self.bndy_ambi()


if __name__ == '__main__':
    """Test the tranp coeff calc."""
    from Mesh import Mesh_1d
    from Plasma1d import Plasma_1d
    mesh1d = Mesh_1d('Plasma_1d', 10e-2, nx=11)
    print(mesh1d)
    plasma1d = Plasma_1d(mesh1d)
    plasma1d.init_plasma()
    # Plasma1d.plot_plasma()
    txp1d = Diff_1d(plasma1d)
    txp1d.calc_transp_coeff(plasma1d)
    txp1d.plot_transp_coeff(plasma1d)
    txp1d.calc_diff(plasma1d)
    txp1d.plot_flux(plasma1d)
    
