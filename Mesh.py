"""
Define Mesh.

Mesh --> 1d or 2d.
"""

from Geometry import Geometry, Rectangle

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

class Mesh(Geometry):
    """Define 1d Mesh."""

    def __init__(self, label, width, height, nx=11, nz=11):
        """Add option to choose nx or res for mesh."""
        super().__init__(label, width, height)
        self.nx = nx
        self.nz = nz
        self.delx = self.width/(self.nx-1)
        self.delz = self.height/(self.nz-1)
        self.x = np.linspace(0.0, self.width, self.nx)

    def __str__(self):
        """Print 1d mesh information."""
        res = 'Mesh_1d:'
        res += '\n ' + super().__str__()
        res += f'\nnx = {self.nx}'
        res += f'\ndelx = {self.delx} m'
        return res

    def add_bndy(self):
        """Add boundaries."""
        pass

    def add_mat(self):
        """Add materials."""
        pass

    def plot_mesh(self):
        """Plot 1d mesh in X."""
        fig, ax = plt.subplots(1, 1, figsize=(4, 4),
                               constrained_layout=True)
        y = np.zeros_like(self.x)
        ax.plot(self.x, y, 'o')
        plt.show(fig)

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


if __name__ == '__main__':
    """Test Mesh."""
    icp2d = Geometry(name='2D ICP', bl=(-0.5, 0.0), width=1.0, height=2.0, 
                     dim=2, is_cyl=False)
    # top = Rectangle(label='Metal', bottom_left=(0.0, 1.9), up_right=(1.0, 2.0))
    top = Rectangle('Metal', (-0.5, 1.9), (0.5, 2.0))
    icp2d.add_shape(top)
    bott = Rectangle('Metal', (-0.5, 0.0), (0.5, 0.1))
    icp2d.add_shape(bott)
    left = Rectangle('Metal', (-0.5, 0.0), (-0.4, 2.0))
    icp2d.add_shape(left)
    right = Rectangle('Metal', (0.4, 0.0), (0.5, 2.0))
    icp2d.add_shape(right)
    icp2d.plot()
    print(icp2d)
    
