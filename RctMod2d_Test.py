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
colMap = copy(cm.get_cmap("jet"))
colMap.set_under(color='white')


if __name__ == '__main__':
    """Test Plasma2d."""
    from RctMod2d_Mesh import Mesh2d
    from RctMod2d_Transp import Diff2d, Ambi2d
    from RctMod2d_React import React_2d
    from RctMod2d_Geom import Geom2d, Domain, Rectangle
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
    pla2d.plot_Te(figsize=figsize, ihoriz=ihoriz)
    
    # init Transp module
    # txp2d = Diff_2d(pla2d)
    txp2d = Ambi2d(pla2d)
    txp2d.calc_transp_coeff(pla2d)
    # txp2d.plot_transp_coeff(pla2d)
    # init Eergy module
    een2d = Eergy2d(pla2d)
    # init React module
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
            txp2d.plot_flux(pla=pla2d, fname=f'flux_itn{itn+1}',
                            figsize=figsize, ihoriz=ihoriz)
    fig = plt.figure(figsize=(4, 4), dpi=300)
    plt.plot(time, ne_ave, 'b-')
    plt.plot(time, ni_ave, 'r-')
    plt.legend(['E', 'Ion'])
    plt.xlabel('Time (s)')
    plt.ylabel('Ave. Density (m^-3)')
    fig.savefig('Density_vs_Time.png', dpi=300)
