"""
Temporary Mesh
"""

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

class Mesh():
    """Define 2d Mesh."""

    def __init__(self, bl=(0.0, 0.0), domain=(1.0, 1.0), ngrid=(11, 11)):
        self.domain = np.asarray(domain)
        self.ngrid = np.asarray(ngrid)
        self.res = np.divide(self.domain, self.ngrid - 1)
        self.width, self.height = self.domain
        self.nx, self.nz = self.ngrid
        self.delx, self.delz = self.res
        tempx = np.linspace(0.0, self.width, self.nx)
        tempz = np.linspace(0.0, self.height, self.nz)
        self.x, self.z = np.meshgrid(tempx, tempz)

    def add_bndy(self):
        """Add boundaries."""
        pass

    def add_mat(self):
        """Add materials."""
        pass

    def plot(self, figsize=(8, 8), dpi=600, fname='Mesh.png'):
        """Plot mesh and surface."""

        fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi,
                                 constrained_layout=True)
        ax = axes[0]
        ax.scatter(self.x, self.z, c=self.mat, s=1, cmap=colMap, vmin=0.2)
        ax = axes[1]
        ax.scatter(self.x, self.z, c=self.surf, s=1)
        fig.savefig(fname, dpi=dpi)

    def cnt_diff(self, y):
        """
        Caculate dy/dx using central differencing.
        
        input: y
        dy/dx = (y[i+1] - y[i-1])/(2.0*dx)
        dy[0] = dy[1]; dy[-1] = dy[-2]
        output: dy
        """
        dy = np.zeros_like(self.x)
        # Although dy[0] and dy[-1] are signed here,
        # they are eventually specified in boundary conditions
        # dy[0] = dy[1]; dy[-1] = dy[-2]
        for i in range(1, self.nx-1):
            dy[i] = (y[i+1] - y[i-1])/self.delx/2.0
        dy[0], dy[-1] = deepcopy(dy[1]), deepcopy(dy[-2])
        return dy
    
    def cnt_diff_2nd(self, y):
        """
        Caculate d2y/dx2 using 2nd order central differencing.

        input: y
        d2y/dx2 = (y[i+1] - 2 * y[i] + y[i-1])/dx^2
        d2y[0] = d2y[1]; d2y[-1] = d2y[-2]
        output: d2y/dx2
        """
        d2y = np.zeros_like(self.x)
        # Although dy[0] and dy[-1] are signed here,
        # they are eventually specified in boundary conditions
        # d2y[0] = d2y[1]; d2y[-1] = d2y[-2]
        for i in range(1, self.nx-1):
            d2y[i] = (y[i+1] - 2 * y[i] + y[i-1])/self.delx**2
        d2y[0], d2y[-1] = deepcopy(d2y[1]), deepcopy(d2y[-2])
        return d2y


