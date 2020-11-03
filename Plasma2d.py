"""
2D Plasma Module - Main Module.

Plasma_2d contains:
    density: update using explicit method, integrating by delt
    temperature: copy from energy module    
"""

from Constants import AMU

import numpy as np
from copy import copy, deepcopy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
colMap = copy(cm.get_cmap("Accent"))
colMap.set_under(color='white')


class Plasma_2d(object):
    """Define 2d Plasma."""

    def __init__(self, geom):
        """
        Plasma_2d is defined as a container.

        Plasma_2d as a basket containing:
            geometry
            physics.
        """
        self.geom = deepcopy(geom)

    def init_plasma(self, ne=1e17, press=10, Te=1, Ti=0.1, Mi=40):
        """
        Initiate plasma attributes.

        ne: 1/m^3, eon denisty
        ni: 1/m^3, ion density = eon density initially
        press: mTorr, pressure
        nn: 1/m^3, neutral density determined by pressure
            At 1 atm, number density = 0.025e27 m^-3.
            At 1 Torr, number density = 3.3e22 m^-3.
            At 1 mTorr, number density = 3.3e19 m^-3.
        Te: eV, eon temperature
        Ti: eV, ion temperature
        coll_e,im: 1/s, coll freq (momentum)
                coll_e,im = 1e7 at 10 mTorr
                            1e8 at 100 mTorr
                            1e9 at 1000 mTorr
        Mi: kg, ion mass
        """
        _x = self.geom.x
        self.ne = np.ones_like(_x)*ne  # init uniform ne on 1d mesh
        self.ni = np.ones_like(_x)*ne  # init ni to neutralize ne
        self.nn = np.ones_like(_x)*(press*3.3e19)  # init neutral density
        self.press = press
        self.Te = np.ones_like(_x)*Te  # init eon temperature
        self.Ti = np.ones_like(_x)*Ti  # init ion temperature
        self.coll_em = np.ones_like(_x)*(press/10.0*1e7)  # eon coll freq (mom)
        self.coll_im = np.ones_like(_x)*(press/10.0*1e7)  # ion coll freq (mom)
        self.Mi = Mi*AMU # ion mass 
        self.set_bc()
        self.limit_plasma()

    def set_bc(self):
        """Impose b.c. on the plasma."""
        for idx in self.geom.bndy_list:
            self.ne[idx] = 1e11
            self.ni[idx] = 1e11
            self.nn[idx] = 1e11
            self.Te[idx] = 0.1
            self.Ti[idx] = 0.01

    def limit_plasma(self, n_min=1e11, n_max=1e22, T_min=0.001, T_max=100.0):
        """Limit variables in the plasma."""
        self.ne = np.clip(self.ne, n_min, n_max)
        self.ni = np.clip(self.ni, n_min, n_max)
        self.nn = np.clip(self.nn, n_min, n_max)
        self.Te = np.clip(self.Te, T_min, T_max)
        self.Ti = np.clip(self.Ti, T_min, T_max)

    def plot_plasma(self, figsize=(8, 8), dpi=600, fname='Plasma.png',
                    imode='Contour'):
        """
        Plot plasma variables vs. position x.

        density, flux, temperature
        imode: str, var, ['Contour', 'Scatter']
        """
        _x, _z = self.geom.x, self.geom.z
        fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi,
                                 constrained_layout=True)
        
        levels = np.linspace(1e11, 1e17, 11)
        
        # plot densities
        if imode == 'Contour':
            ax = axes[0]
            ax.contourf(_x, _z, self.ne, cmap='rainbow', levels=levels)
            ax.set_title('E Density')
            
            ax = axes[1]
            ax.contourf(_x, _z, self.ni, cmap='rainbow', levels=levels)
            ax.set_title('Ion Density')
            
        elif imode == 'Scatter':
            ax = axes[0]
            ax.scatter(_x, _z, c=self.ne, s=1, cmap=colMap, vmin=0.2)
            ax.set_title('E Density')
            
            ax = axes[1]
            ax.scatter(_x, _z, c=self.ni, s=1, cmap=colMap, vmin=0.2)
            ax.set_title('Ion Density')
            
        for ax in axes:
            ax.set_xlabel('Position (m)')
            ax.set_ylabel('Height (m)')
            ax.set_aspect('equal')
        fig.savefig(fname, dpi=dpi)

    def init_pot(self, phi=0.0):
        """Initiate potential attributes."""
        nx = self.geom.nx
        self.pot = np.ones(nx)*phi  # initial uniform potential
        self.ef = np.zeros_like(self.pot)  # initial uniform E-field
        self.ef_ambi = np.zeros_like(self.pot)  # initial ambipolar E-field

    def plot_pot(self):
        """Plot potential, E-field."""
        x = self.geom.x
        fig, axes = plt.subplots(1, 2, figsize=(8, 4),
                                 constrained_layout=True)
        # plot potential
        ax = axes[0]
        ax.plot(x, self.pot, 'y-')
        ax.legend(['Potential'])
        # plot E-field
        ax = axes[1]
        ax.plot(x, self.ef, 'g-')
        ax.legend(['E-field'])
        plt.show()

    def den_evolve(self, delt, txp, src):
        """
        Evolve the density in Plasma by solving the continuity equation.

        dn/dt = -dFlux/dx + Se
        dn(t + dt) = dn(t) - dFlux/dx*dt + Se*dt
        delt: time step for explict method
        txp: object for transport module
        src: object for reaction module
        """
        self.ne += (-txp.dfluxe + src.se)*delt
        self.ni += (-txp.dfluxi + src.si)*delt


if __name__ == '__main__':
    """Test Plasma_1d."""
    from Mesh_temp import Mesh
    from Transp2d import Diff_2d
    from React2d import React_2d
    domain=(2.0, 4.0)
    mesh2d = Mesh(bl=(-1.0, 0.0), domain=domain, ngrid=(21, 41))
    mesh2d.find_bndy()
    mesh2d.plot()
    
    pla2d = Plasma_2d(mesh2d)
    pla2d.init_plasma()
    pla2d.set_bc()
    figsize = tuple([domain[0]*2, domain[1]])
    pla2d.plot_plasma(figsize=figsize)
    
    # calc the transport 
    txp2d = Diff_2d(pla2d)
    txp2d.calc_transp_coeff(pla2d)
    # txp2d.plot_transp_coeff(pla2d)
    # calc source term
    src2d = React_2d(pla2d)
    # #
    # ne_ave, ni_ave = [], []
    # time = []
    dt = 1e-8
    niter = 3000
    for itn in range(niter):
        txp2d.calc_diff(pla2d)
        pla2d.den_evolve(dt, txp2d, src2d)
        pla2d.set_bc()
        pla2d.limit_plasma()
        # ne_ave.append(np.mean(pla2d.ne))
        # ni_ave.append(np.mean(pla2d.ni))
        # time.append(dt*(niter+1))
        if not (itn+1) % (niter/10):
            # txp2d.plot_flux(pla2d)
            pla2d.plot_plasma(fname=f'plamat_itn{itn+1}', figsize=figsize)
            # pla2d.plot_plasma()
