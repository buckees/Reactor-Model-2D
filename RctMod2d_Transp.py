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
from copy import copy, deepcopy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
colMap = copy(cm.get_cmap("Accent"))
colMap.set_under(color='white')



class Transp_2d(object):
    """Define the base tranport module/object."""
    
    def __init__(self, pla):
        """Import geometry information."""
        _x = pla.mesh.x
        self.fluxex = np.zeros_like(_x)  # initial eon flux in x direction
        self.fluxez = np.zeros_like(_x)  # initial eon flux in z direction
        self.fluxix = np.zeros_like(_x)  # initial ion flux in x direction
        self.fluxiz = np.zeros_like(_x)  # initial ion flux in z direction
        self.dfluxe = np.zeros_like(_x)  # initial eon flux
        self.dfluxi = np.zeros_like(_x)  # initial ion flux

    def calc_transp_coeff(self, pla):
        """
        Calc diffusion coefficient and mobility.

        pla: Plasma2d object/class
        De,i: m^2/s, (nz, nx) matrix, D = k*T/(m*coll_m)
        Mue,i: m^2/(V*s), (nz, nx) matrix, Mu = q/(m*coll_m)
        D and Mu depend only on pla.
        """
        # calc diff coeff: D = k*T/(m*coll_m)
        self.De = np.divide(KB_EV*pla.Te, EON_MASS*pla.coll_em)  
        self.Di = np.divide(KB_EV*pla.Ti, pla.Mi*pla.coll_im)  
        # calc mobility: Mu = q/(m*coll_m)
        self.Mue = UNIT_CHARGE/EON_MASS/pla.coll_em
        self.Mui = UNIT_CHARGE/pla.Mi/pla.coll_im

    def plot_transp_coeff(self, pla, 
                          figsize=(8, 8), dpi=300, fname='Transp_coeff.png'):
        """
        Plot transp coeff.
        
        pla: Plasma2d object
            use pla.mesh.x for plot
        """
        _x, _z = pla.mesh.x, pla.mesh.z
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

    
    def plot_flux(self, pla,
                  figsize=(8, 8), dpi=600, fname='Flux.png'):
        """
        Plot flux and dflux.
        
        pla: Plasma_1d object
            use pla.mesh.x for plot
        """
        _x, _z = pla.mesh.x, pla.mesh.z
        fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi,
                                 constrained_layout=True)
        # plot densities
        ax = axes[0]
        ax.scatter(_x, _z, c=self.fluxe, s=1, cmap=colMap, vmin=0.2)
        ax.set_title('E Flux')
        ax.set_xlabel('Position (m)')
        ax.set_ylabel('Height (m)')
        ax = axes[1]
        ax.scatter(_x, _z, c=self.fluxi, s=1, cmap=colMap, vmin=0.2)
        ax.set_title('Ion Flux')
        ax.set_xlabel('Position (m)')
        ax.set_ylabel('Height (m)')
        fig.savefig(fname, dpi=dpi)


class Diff_2d(Transp_2d):
    """
    Calc the dflux for Diffusion Only Module.
    
    dn/dt = -D * d2n/dx2 + Se
    D: m^2/s, diffusion coefficient is calc from Tranps_1d
    Output: D * d2n/dx2
    """
    
    def calc_diff(self, pla):
        """Calc diffusion term: D * d2n/dx2 and diffusion flux D * dn/dx."""
        # Calc transp coeff first
        self.calc_transp_coeff(pla)
        # Calc flux
        self.fluxex, self.fluxez = -self.De * pla.mesh.cnt_diff(pla.ne)
        self.fluxix, self.fluxiz = -self.Di * pla.mesh.cnt_diff(pla.ni)
        # Calc dflux
        self.dfluxe = -self.De * pla.mesh.cnt_diff_2nd(pla.ne)
        self.dfluxi = -self.Di * pla.mesh.cnt_diff_2nd(pla.ni)

    
class Ambi_2d(Transp_2d):
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
        _dnix, _dniz = pla.mesh.cnt_diff(pla.ni)
        self.Eax = np.divide(self.Di - self.De, self.Mui + self.Mue)
        self.Eaz = deepcopy(self.Eax)
        self.Eax *= np.divide(_dnix, pla.ni)
        self.Eaz *= np.divide(_dniz, pla.ni)
        # # Calc flux
        self.fluxex, self.fluxez = -self.Da * pla.mesh.cnt_diff(pla.ne)
        self.fluxix, self.fluxiz = -self.Da * pla.mesh.cnt_diff(pla.ni)
        # Calc dflux
        self.dfluxe = -self.Da * pla.mesh.cnt_diff_2nd(pla.ne)
        self.dfluxi = -self.Da * pla.mesh.cnt_diff_2nd(pla.ni)
        # self.bndy_ambi()


if __name__ == '__main__':
    """Test the tranp coeff calc."""
    from Mesh_temp import Mesh
    from Plasma2d import Plasma_2d
    mesh2d = Mesh(bl=(-1.0, 0.0), domain=(2.0, 4.0), ngrid=(21, 41))
    mesh2d.find_bndy()
    mesh2d.plot()
    
    pla2d = Plasma_2d(mesh2d)
    pla2d.init_plasma()
    pla2d.plot_plasma()

    txp2d = Diff_2d(pla2d)
    txp2d.calc_transp_coeff(pla2d)
    # txp2d.plot_transp_coeff(pla2d)
    txp2d.calc_diff(pla2d)
    # txp2d.plot_flux(pla2d)
    
    # from Mesh import Mesh_1d
    # from Plasma1d import Plasma_1d
    # mesh1d = Mesh_1d('Plasma_1d', 10e-2, nx=11)
    # print(mesh1d)
    # plasma1d = Plasma_1d(mesh1d)
    # plasma1d.init_plasma()
    # # Plasma1d.plot_plasma()
    # txp1d = Diff_1d(plasma1d)
    # txp1d.calc_transp_coeff(plasma1d)
    # txp1d.plot_transp_coeff(plasma1d)
    # txp1d.calc_diff(plasma1d)
    # txp1d.plot_flux(plasma1d)
    
