"""
Geometry module.

Geometry module defines the basic shapes, such as 
1D: Interval 
2D: Rectangle, Triangle, etc.

Geometry module defines the construction of the geometry.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patch
import numpy as np

class Shape():
    """Init the Shape."""
    
    def __init__(self, label):
        """
        Init the Shape.
        
        label: str, var, label of shape.
        """
        self.label = label

    def __str__(self):
        """Print Shape info."""
        return f'label = {self.label}'

class Domain(Shape):
    """Define the Domain."""
    
    def __init__(self, name='Domain', bl=(0.0, 0.0), domain=(1.0, 1.0)):
        """
        Init the Domain.
        
        name: str, var, name of domain.
        bl: unit in m, (2, ) tuple
        domain: unit in m, (2, ) tuple, width and height
        """
        self.name = name
        self.bl = np.asarray(bl)
        self.domain = np.asarray(domain)
        super().__init__(label='Plasma')

    def __str__(self):
        """Print Domain info."""
        res = 'Domain:'
        res += f'\nname = {self.name}'
        res += f'\nlabel = {self.label}'
        res += f'\nbottom left = {self.bl} m'
        res += f'\ndomain = {self.domain} m'
        return res

class Rectangle(Shape):
    """Rectangle is a 2D basic shape."""
    
    def __init__(self, label, bottom_left, up_right):
        """
        Init the Rectangle.
        
        bottom_left: unit in m, (2, ) tuple
        up_right: unit in m, (2, ) tuple
        """
        super().__init__(label)
        self.bl = np.asarray(bottom_left)
        self.ur = np.asarray(up_right)
        self.width = self.ur[0] - self.bl[0]
        self.height = self.ur[1] - self.bl[1]
        self.type = 'Rectangle'

    def __str__(self):
        """Print Rectangle info."""
        res = 'Rectangle:'
        res += f'\nbottom left = {self.bl} m'
        res += f'\nup right = {self.ur} m'
        return res

    def __contains__(self, posn):
        """
        Determind if a position is inside the Interval.
        
        posn: unit in m, (2, ) array, position as input
        boundaries are not consindered as "Inside"
        """
        return all(self.bl < posn < self.ur)


class Geom2d():
    """Constuct the geometry."""
    
    def __init__(self, name='Geometry', is_cyl=False):
        """
        Init the geometry.
        
        dim = 2, this class only supports 2D geometry
        is_cyl: bool, wether the geometry is cylidrical symmetric or not
        """
        self.name = name
        self.dim = 2
        self.is_cyl = is_cyl
        self.sequence = list()

    def __str__(self):
        """Print Geometry info."""
        res = f'Geometry dimension {self.dim}D'
        if self.is_cyl:
            res += ' cylindrical'
        res += '\nGeometry sequence:'
        for shape in self.sequence:
            res += '\n' + str(shape)
        return super(Domain, self).__str__() + res

    def add_domain(self, domain):
        """
        Add domain to the geometry.
        
        domain: class
        """
        pass

    def add_shape(self, shape):
        """
        Add shape to the geometry.
        
        shape: class
        2D - shape is an instance of Rectangle()
        """
        self.sequence.append(shape)

    def get_label(self, posn):
        """
        Return the label of a position.
        
        posn: unit in m, var or (2, ) array, position as input
        label: str, var, label of the shape
        """
        # what if return None
        label = None
        for shape in self.sequence:
            if posn in shape:
                label = shape.label
        return label

    def label_check(self, posn, label):
        """
        Check if labelf of posn == label of input.
        
        posn: unit in m, var or (2, ) array, position as input
        label: str, var, label as input
        """
        # cannot determined if the posn is in domain
        res = False
        posn_label = self.get_label(posn)
        return res or (posn_label == label)
    
    def plot(self, figsize=(8, 8), dpi=300, fname='Geometry'):
        """
        Plot the geometry.
        
        figsize: unit in inch, (2, ) tuple, determine the fig/canvas size
        dpi: dimless, int, Dots Per Inch
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi,
                         constrained_layout=True)
        ax = axes[0]
        
        for shape in self.sequence:
            if shape.type == 'Rectangle':
                ax.add_patch(
                    patch.Rectangle(shape.bl, shape.width, shape.height,
                                    facecolor='k', edgecolor='b'))
        ax = axes[1]
        ax.add_patch(
            patch.Rectangle(self.bl, self.width, self.height, 
                            facecolor='b'))
        for shape in self.sequence:
            if shape.type == 'Rectangle':
                ax.add_patch(
                    patch.Rectangle(shape.bl, shape.width, shape.height,
                                    facecolor='w', edgecolor='w'))
        for ax in axes:
            ax.set_xlim(self.bl[0], self.bl[0] + self.width)
            ax.set_ylim(self.bl[1], self.bl[1] + self.height)
        fig.savefig(fname, dpi=dpi)
                
if __name__ == '__main__':
    icp2d = Geometry(name='2D ICP', bl=(-1.0, 0.0), width=2.0, height=4.0, 
                     dim=2, is_cyl=False)
    top = Rectangle('Metal', (-1.0, 3.5), (1.0, 4.0))
    icp2d.add_shape(top)
    bott = Rectangle('Metal', (-0.8, 0.0), (0.8, 0.2))
    icp2d.add_shape(bott)
    left = Rectangle('Metal', (-1.0, 0.0), (-0.9, 4.0))
    icp2d.add_shape(left)
    right = Rectangle('Metal', (0.9, 0.0), (1.0, 4.0))
    icp2d.add_shape(right)
    icp2d.plot(fname='ICP2d.png')
    print(icp2d)
