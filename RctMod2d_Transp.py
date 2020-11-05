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
colMap = copy(cm.get_cmap("jet"))
colMap.set_under(color='white')



class Transp2d(object):
    """Define the base tranport module/object."""
    
    def __init__(self, pla):
        """Import geometry information."""
        self.fluxex = np.zeros_like(pla.ne)  # initial eon flux in x direction
        self.fluxez = np.zeros_like(pla.ne)  # initial eon flux in z direction
        self.fluxix = np.zeros_like(pla.ne)  # initial ion flux in x direction
        self.fluxiz = np.zeros_like(pla.ne)  # initial ion flux in z direction
        self.dfluxe = np.zeros_like(pla.ne)  # initial eon flux
        self.dfluxi = np.zeros_like(pla.ne)  # initial ion flux

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

    def plot_transp_coeff(self, pla, figsize=(8, 8), ihoriz=1, 
                    dpi=300, fname='Transp_coeff.png', imode='Contour'):
        """
        Plot transp coeff vs position.
        
        var include diffusion coeff and mobility.
        pla: Plasma2d object
            use pla.mesh.x,z for plot
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
        # plot transp coeff
        if imode == 'Contour':
            for _ax, _den, _title in zip(axes, (self.De, self.Di), 
                            ('E Diffusion Coeff', 'Ion Diffusion Coeff')):
                _cs = _ax.contourf(_x, _z, _den, cmap=colMap)
                _ax.set_title(_title)
                fig.colorbar(_cs, ax=_ax, shrink=0.9)
            
        elif imode == 'Scatter':
            for _ax, _den, _title in zip(axes, (self.De, self.Di), 
                                         ('E Density', 'Ion Density')):
                _ax.scatter(_x, _z, c=_den, s=5, cmap=colMap)
                _ax.set_title(_title)
            
        for ax in axes:
            ax.set_xlabel('Position (m)')
            ax.set_ylabel('Height (m)')
            ax.set_aspect('equal')
        fig.savefig(fname, dpi=dpi)
        plt.close()
    
    def plot_flux(self, pla, figsize=(8, 8), ihoriz=1, 
                    dpi=300, fname='flux.png', imode='Contour'):
        """
        Plot flux vs position.
        
        var include flux and dflux.
        pla: Plasma2d object
            use pla.mesh.x,z for plot
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
        # plot flux
        if imode == 'Contour':
            for _ax, _den, _title in zip(axes, (self.fluxex, self.fluxez), 
                            ('E flux in x', 'E flux in z')):
                _cs = _ax.contourf(_x, _z, _den, cmap=colMap)
                _ax.set_title(_title)
                fig.colorbar(_cs, ax=_ax, shrink=0.9)
        
        for ax in axes:
            ax.set_xlabel('Position (m)')
            ax.set_ylabel('Height (m)')
            ax.set_aspect('equal')
        fig.savefig('Eon_'+fname, dpi=dpi)
        plt.close()
        
        # plot dflux
        if ihoriz:
            fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi,
                                     constrained_layout=True)
        else:
            fig, axes = plt.subplots(2, 1, figsize=figsize, dpi=dpi,
                                     constrained_layout=True)
        # plot flux
        if imode == 'Contour':
            for _ax, _den, _title in zip(axes, (self.dfluxe, self.dfluxi), 
                            ('E dflux', 'Ion dflux')):
                _cs = _ax.contourf(_x, _z, _den, cmap=colMap)
                _ax.set_title(_title)
                fig.colorbar(_cs, ax=_ax, shrink=0.9)
        
        for ax in axes:
            ax.set_xlabel('Position (m)')
            ax.set_ylabel('Height (m)')
            ax.set_aspect('equal')
        fig.savefig('d'+fname, dpi=dpi)
        plt.close()


class Diff2d(Transp2d):
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

    
class Ambi2d(Transp2d):
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
    from RctMod2d_Mesh import Mesh2d
    from RctMod2d_React import React_2d
    from RctMod2d_Geom import Geom2d, Domain, Rectangle
    from RctMod2d_Plasma import Plasma2d
    # build the geometry
    geom2d = Geom2d(name='2D Plasma', is_cyl=False)
    domain = Domain((-1.0, 0.0), (2.0, 4.0))
    geom2d.add_domain(domain)
    top = Rectangle('Metal', (-1.0, 3.5), (1.0, 4.0))
    geom2d.add_shape(top)
    bott = Rectangle('Metal', (-0.5, 0.0), (0.5, 1.0))
    geom2d.add_shape(bott)
    left = Rectangle('Metal', (-1.0, 0.0), (-0.9, 4.0))
    geom2d.add_shape(left)
    right = Rectangle('Metal', (0.9, 0.0), (1.0, 4.0))
    geom2d.add_shape(right)
    quartz = Rectangle('Quartz', (-0.9, 3.3), (0.9, 3.5))
    geom2d.add_shape(quartz)
    geom2d.plot(fname='geom2d.png')
    print(geom2d)
    # generate mesh to imported geometry
    mesh2d = Mesh2d()
    mesh2d.import_geom(geom2d)
    mesh2d.generate_mesh(ngrid=(21, 41))
    mesh2d.plot()

    
    pla2d = Plasma2d(mesh2d)
    pla2d.init_plasma()

    if domain.domain[0] > domain.domain[1]:
        figsize = tuple([domain.domain[0], domain.domain[1]*2])
        ihoriz = 0
    else:
        figsize = tuple([domain.domain[0]*2*1.5, domain.domain[1]])
        ihoriz = 1
    pla2d.plot_plasma(figsize=figsize, ihoriz=ihoriz)
    
    txp2d = Ambi2d(pla2d)
    txp2d.calc_transp_coeff(pla2d)
    txp2d.plot_transp_coeff(pla=pla2d, figsize=figsize, ihoriz=ihoriz)
    txp2d.calc_ambi(pla2d)
    txp2d.plot_flux(pla=pla2d, figsize=figsize, ihoriz=ihoriz)
    
