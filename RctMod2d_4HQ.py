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

from Constants import PI
import numpy as np
from copy import copy
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
#               (left, bottom), (width, height)
domain = Domain((-0.25, 0.0),    (0.5, 0.4))
geom2d.add_domain(domain)

# Add metal wall to all boundaries
# In Metal, vector potential A = 0
#                        (left, bottom), (right, top)
top = Rectangle('Metal', (-0.25, 0.38), (0.25, 0.4))
geom2d.add_shape(top)
bott = Rectangle('Metal', (-0.25, 0.0), (0.25, 0.02))
geom2d.add_shape(bott)
# use -0.231 instead of -0.23 for mesh asymmetry
left = Rectangle('Metal', (-0.25, 0.0), (-0.231, 0.4)) 
geom2d.add_shape(left)
right = Rectangle('Metal', (0.23, 0.0), (0.25, 4.0))
geom2d.add_shape(right)
ped = Rectangle('Metal', (-0.20, 0.0), (0.20, 0.1))
geom2d.add_shape(ped)


# Add quartz to separate coil area and plasma area
# Quartz conductivity = 1e-5 S/m
quartz = Rectangle('Quartz', (-0.23, 0.3), (0.23, 0.32))
geom2d.add_shape(quartz)

# Add air to occupy the top coil area to make it non-plasma
# Air concudctivity = 0.0 S/m
air = Rectangle('Air', (-0.23, 0.32), (0.23, 0.38))
geom2d.add_shape(air)

# Add coil within air and overwirte air
# coil 1, 2, 3: J = -J0*exp(iwt)
# coil 4, 5, 6: J = +J0*exp(iwt)
coil1 = Rectangle('Coil', (-0.20, 0.34), (-0.18, 0.36))
geom2d.add_shape(coil1)
coil2 = Rectangle('Coil', (-0.14, 0.34), (-0.12, 0.36))
geom2d.add_shape(coil2)
coil3 = Rectangle('Coil', (-0.08, 0.34), (-0.06, 0.36))
geom2d.add_shape(coil3)
coil4 = Rectangle('Coil', (0.18, 0.34), (0.20, 0.36))
geom2d.add_shape(coil4)
coil5 = Rectangle('Coil', (0.12, 0.34), (0.14, 0.36))
geom2d.add_shape(coil5)
# use 0.081 instead of 0.08 for mesh asymmetry
coil6 = Rectangle('Coil', (0.06, 0.34), (0.081, 0.36))
geom2d.add_shape(coil6)



geom2d.plot(figsize=(10, 4), ihoriz=1)
print(geom2d)
# generate mesh to imported geometry
mesh2d = Mesh2d()
mesh2d.import_geom(geom2d)
mesh2d.generate_mesh(ngrid=(51, 41))
mesh2d.plot(figsize=(10, 4), ihoriz=1)


pla2d = Plasma2d(mesh2d)
pla2d.init_plasma(ne=1e16, Te=1.5)

pla2d.get_eps()
pla2d.plot_eps()

temp_ratio = domain.domain[0]/domain.domain[1]
if temp_ratio > 2.0:
    figsize = (4, 8)
    ihoriz = 0
else:
    figsize = (10, 4)
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
pwr2d.calc_pwr_in(pla2d, pwr=0.0, imode='ne')
pwr2d.plot_pwr(pla2d)
# init Eergy module
een2d = Eergy2d(pla2d)
een2d.get_pwr(pwr2d)

pla2d.plot_plasma(fname='plasma_itn0', 
                  figsize=figsize, ihoriz=ihoriz,
                  iplot_geom=0)

dt = 1e-5
niter = 100
for itn in range(niter):
    txp2d.calc_ambi(pla2d)
    pla2d.den_evolve(dt, txp2d, src2d)
    if not (itn+1) % (niter/10):
        # txp2d.plot_flux(pla2d)
        pla2d.plot_plasma(fname=f'plasma_itn{itn+1}', 
                          figsize=figsize, ihoriz=ihoriz)
        txp2d.plot_flux(pla=pla2d, fname=f'flux_itn{itn+1}',
                        figsize=figsize, ihoriz=ihoriz)


ne_ave, ni_ave, Te_ave = [], [], []
time = []
niter = 10000
dt = 1e-5
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
        een2d.plot_dQe(pla=pla2d, fname=f'dQe_itn{itn+1}', 
                          figsize=figsize, ihoriz=ihoriz)
        src2d.plot_src(pla=pla2d, fname=f'src_itn{itn+1}', 
                          figsize=figsize, ihoriz=ihoriz)
        txp2d.plot_flux(pla=pla2d, fname=f'flux_itn{itn+1}',
                        figsize=figsize, ihoriz=ihoriz)
        pla2d.calc_conde(2*PI*13.56e6)
        pla2d.plot_conde(fname=f'conde_itn{itn+1}', 
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

# save matrix in numpy binary format
np.save('cond_e', pla2d.conde)
np.save('cond_e_real', pla2d.conde.real)
np.save('cond_e_imag', pla2d.conde.imag)

# save matrix in csv format
np.savetxt("cond_e.csv", pla2d.conde, delimiter=",")
np.savetxt("cond_e_real.csv", pla2d.conde.real, delimiter=",")
np.savetxt("cond_e_imag.csv", pla2d.conde.imag, delimiter=",")
