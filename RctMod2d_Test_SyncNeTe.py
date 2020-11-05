"""
2D Plasma Module - Main Module.

Plasma_2d contains:
    density: update using explicit method, integrating by delt
    temperature: copy from energy module    
"""

import os
import glob
for i in glob.glob("*.png"):
    os.remove(i)

from Constants import AMU
import numpy as np
from copy import copy, deepcopy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
colMap = copy(cm.get_cmap("jet"))
colMap.set_under(color='white')


"""Import plasma modules."""
from RctMod2d_Geom import Geom2d, Domain, Rectangle
from RctMod2d_Mesh import Mesh2d
from RctMod2d_Plasma import Plasma2d
from RctMod2d_Transp import Ambi2d
from RctMod2d_React import React2d
from RctMod2d_Eergy import Eergy2d
from RctMod2d_Power import Power2d

# build the geometry
geom2d = Geom2d(name='2D Plasma', is_cyl=False)
domain = Domain((-1.0, 0.0), (2.0, 4.0))
geom2d.add_domain(domain)
top = Rectangle('Metal', (-1.0, 3.5), (1.0, 4.0))
geom2d.add_shape(top)
bott = Rectangle('Metal', (-0.3, 0.0), (0.3, 1.0))
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
pla2d.init_plasma(ne=1e16, Te=1.5)

if domain.domain[0] > domain.domain[1]:
    figsize = tuple([domain.domain[0], domain.domain[1]*2])
    ihoriz = 0
else:
    figsize = tuple([domain.domain[0]*2*1.5, domain.domain[1]])
    ihoriz = 1
pla2d.plot_plasma(figsize=figsize, ihoriz=ihoriz)
pla2d.plot_Te(figsize=figsize, ihoriz=ihoriz)

# init Transp module
# txp2d = Diff_2d(pla2d)
txp2d = Ambi2d(pla2d)
txp2d.calc_transp_coeff(pla2d)
txp2d.calc_ambi(pla2d)
# txp2d.plot_transp_coeff(pla2d)
# init React module
src2d = React2d(pla2d)
src2d.calc_src(pla2d)
src2d.plot_src(pla=pla2d, fname='src_itn0', 
                          figsize=figsize, ihoriz=ihoriz)
# init Power module
pwr2d = Power2d(pla2d)
pwr2d.calc_pwr_in(pla2d, pwr=10.0, imode='ne')
# init Eergy module
een2d = Eergy2d(pla2d)
een2d.get_pwr(pwr2d)

pla2d.plot_plasma(fname='plasma_itn0', 
                          figsize=figsize, ihoriz=ihoriz)

dt = 1e-3
niter = 30
for itn in range(niter):
    txp2d.calc_ambi(pla2d)
    pla2d.den_evolve(dt, txp2d, src2d)
    if not (itn+1) % (niter/3):
        # txp2d.plot_flux(pla2d)
        pla2d.plot_plasma(fname=f'plasma_itn{itn+1}', 
                          figsize=figsize, ihoriz=ihoriz)
        txp2d.plot_flux(pla=pla2d, fname=f'flux_itn{itn+1}',
                        figsize=figsize, ihoriz=ihoriz)


ne_ave, ni_ave, Te_ave = [], [], []
time = []
niter = 30000
dt = 1e-4
niter_Te = 30
for itn in range(niter):
    pwr2d.calc_pwr_in(pla2d, pwr=100.0, imode='ne')
    txp2d.calc_ambi(pla2d)
    een2d.get_pwr(pwr2d)
    for itn_Te in range(niter_Te):    
        een2d.calc_Te(dt/niter_Te, pla2d, txp2d)
    pla2d.get_Te(een2d)
    src2d.calc_src(pla2d)
    pla2d.den_evolve(dt, txp2d, src2d)
    ne_ave.append(pla2d.ne.mean())
    ni_ave.append(pla2d.ni.mean())
    Te_ave.append(pla2d.Te.mean())
    time.append(dt*(itn+1))
    if not (itn+1) % (niter/10):
        pla2d.plot_plasma(fname=f'plasma_itn{itn+1}', 
                          figsize=figsize, ihoriz=ihoriz)
        pla2d.plot_Te(fname=f'Te_itn{itn+1}', 
                          figsize=figsize, ihoriz=ihoriz)
        src2d.plot_src(pla=pla2d, fname=f'src_itn{itn+1}', 
                          figsize=figsize, ihoriz=ihoriz)
        txp2d.plot_flux(pla=pla2d, fname=f'flux_itn{itn+1}',
                        figsize=figsize, ihoriz=ihoriz)

        # plot ave. values
        fig, axes = plt.subplots(1, 2, figsize=(8,4), dpi=300,
                                             constrained_layout=True)
        ax = axes[0]
        ax.plot(time, ne_ave, 'b-')
        ax.legend(['ne'])
        ax.set_title('Eon Density (m^-3)')
        plt.xlabel('Time (s)')
        plt.ylabel('Ave. Density (m^-3)')
        
        ax = axes[1]
        ax.plot(time, Te_ave, 'r-')
        ax.legend(['Te'])
        ax.set_title('Eon Temperature (eV)')
        plt.xlabel('Time (s)')
        plt.ylabel('Ave. Eon Temperature (eV)')
        
        fig.savefig('Ave_vs_Time.png', dpi=300)
        plt.close()