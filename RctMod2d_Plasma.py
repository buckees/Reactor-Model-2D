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


class Plasma2d(object):
    """Define 2d Plasma."""

    def __init__(self, mesh):
        """
        Plasma2d is defined as a container.

        Plasma_2d as a basket containing:
            mesh
            physics.
        """
        self.mesh = mesh

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
        _x = self.mesh.x
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
        self.set_nonPlasma()
        self.limit_plasma()

    def set_bc(self):
        """Impose b.c. on the plasma."""
        for _idx in self.mesh.bndy_list:
            self.ne[_idx] = 1e11
            self.ni[_idx] = 1e11
            self.nn[_idx] = 1e11
            self.Te[_idx] = 0.1
            self.Ti[_idx] = 0.01

    def set_nonPlasma(self):
        """Impose fixed values on the non-plasma materials."""
        for _idx, _mat in np.ndenumerate(self.mesh.mat):
            if _mat:
                self.ne[_idx] = 1e11
                self.ni[_idx] = 1e11
                self.nn[_idx] = 1e11
                self.Te[_idx] = 0.1
                self.Ti[_idx] = 0.01

    def limit_plasma(self, n_min=1e11, n_max=1e22, T_min=0.001, T_max=100.0):
        """Limit variables in the plasma."""
        self.ne = np.clip(self.ne, n_min, n_max)
        self.ni = np.clip(self.ni, n_min, n_max)
        self.nn = np.clip(self.nn, n_min, n_max)
        self.Te = np.clip(self.Te, T_min, T_max)
        self.Ti = np.clip(self.Ti, T_min, T_max)

    def plot_plasma(self, figsize=(8, 8), ihoriz=1, 
                    dpi=600, fname='Plasma.png', imode='Contour'):
        """
        Plot plasma variables vs. position x.

        density, flux, temperature
        imode: str, var, ['Contour', 'Scatter']
        """
        _x, _z = self.mesh.x, self.mesh.z
        if ihoriz:
            fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi,
                                     constrained_layout=True)
        else:
            fig, axes = plt.subplots(2, 1, figsize=figsize, dpi=dpi,
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
        nx = self.mesh.nx
        self.pot = np.ones(nx)*phi  # initial uniform potential
        self.ef = np.zeros_like(self.pot)  # initial uniform E-field
        self.ef_ambi = np.zeros_like(self.pot)  # initial ambipolar E-field

    def plot_pot(self):
        """Plot potential, E-field."""
        x = self.mesh.x
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
    from RctMod2d_Mesh import Mesh2d
    from RctMod2d_Transp import Diff_2d, Ambi_2d
    from RctMod2d_React import React_2d
    from RctMod2d_Geom import Geom2d, Domain, Rectangle
    # build the geometry
    geom2d = Geom2d(name='2D Plasma', is_cyl=False)
    domain = Domain((-1.0, 0.0), (2.0, 4.0))
    geom2d.add_domain(domain)
    top = Rectangle('Metal', (-1.0, 3.5), (1.0, 4.0))
    geom2d.add_shape(top)
    bott = Rectangle('Metal', (-0.5, 0.0), (0.5, 0.2))
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
        figsize = tuple([domain.domain[0]*2, domain.domain[1]])
        ihoriz = 1
    pla2d.plot_plasma(figsize=figsize, ihoriz=ihoriz)
    
    # calc the transport 
    # txp2d = Diff_2d(pla2d)
    txp2d = Ambi_2d(pla2d)
    txp2d.calc_transp_coeff(pla2d)
    # txp2d.plot_transp_coeff(pla2d)
    # calc source term
    src2d = React_2d(pla2d)
    #
    ne_ave, ni_ave = [], []
    time = []
    dt = 1e-3
    niter = 3000
    for itn in range(niter):
        txp2d.calc_ambi(pla2d)
        pla2d.den_evolve(dt, txp2d, src2d)
        pla2d.set_bc()
        pla2d.limit_plasma()
        ne_ave.append(pla2d.ne.mean())
        ni_ave.append(pla2d.ni.mean())
        time.append(dt*(itn+1))
        if not (itn+1) % (niter/10):
            # txp2d.plot_flux(pla2d)
            pla2d.plot_plasma(fname=f'plasma_itn{itn+1}', 
                              figsize=figsize, ihoriz=ihoriz)
    fig = plt.figure(figsize=(4, 4), dpi=300)
    plt.plot(time, ne_ave, 'b-')
    plt.plot(time, ni_ave, 'r-')
    plt.legend(['E', 'Ion'])
    plt.xlabel('Time (s)')
    plt.ylabel('Ave. Density (m^-3)')
    fig.savefig('Density_vs_Time.png', dpi=300)
