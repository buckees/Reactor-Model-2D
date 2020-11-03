"""
2D Plasma Module - Main Module

Plasma_2d contains:
    density: update using explicit method, integrating by delt
    temperature: copy from energy module    
"""

from Constants import AMU

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


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
        x = self.geom.x
        self.ne = np.ones(x)*ne  # init uniform ne on 1d mesh
        self.ni = np.ones(x)*ne  # init ni to neutralize ne
        self.nn = np.ones(x)*(press*3.3e19)  # init neutral density
        self.press = press
        self.Te = np.ones(x)*Te  # init eon temperature
        self.Ti = np.ones(x)*Ti  # init ion temperature
        self.coll_em = np.ones(x)*(press/10.0*1e7)  # eon coll freq (mom)
        self.coll_im = np.ones(x)*(press/10.0*1e7)  # ion coll freq (mom)
        self.Mi = Mi*AMU # ion mass 
        self.bndy_plasma()
        self.limit_plasma()

    def bndy_plasma(self):
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

    def plot(self):
        """
        Plot plasma variables vs. position x.

        density, flux, temperature
        """
        x = self.geom.x
        fig, axes = plt.subplots(1, 2, figsize=(8, 3),
                                 constrained_layout=True)
        # plot densities
        ax = axes[0]
        ax.plot(x, self.ne, 'b-')
        ax.plot(x, self.ni, 'r-')
        ax.legend(['E', 'Ion'])
        ax.set_xlabel('Position (m)')
        ax.set_ylabel('Density (m^-3)')
        # plot temperature
        ax = axes[1]
        ax.plot(x, self.Te, 'b-')
        ax.plot(x, self.Ti, 'r-')
        ax.legend(['Te', 'Ti'])
        ax.set_xlabel('Position (m)')
        ax.set_ylabel('Temperature (eV)')
        plt.show()

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
    from Mesh import Mesh_1d
    from Transp1d import Diff_1d, Ambi_1d
    from React1d import React_1d
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
        ne_ave.append(np.mean(pla1d.ne))
        ni_ave.append(np.mean(pla1d.ni))
        time.append(dt*(niter+1))
        if not (itn+1) % (niter/10):
            txp1d.plot_flux(pla1d)
            pla1d.plot_plasma()
