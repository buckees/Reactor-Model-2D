"""
This is the Geometry module.

Geometry module defines the basic shapes, such as 
1D: Interval 
2D: Rectangle, Triangle, etc.

Geometry module defines the construction of the geometry.
"""


class Shape:
    def __init__(self, label):
        self.label = label

    def __str__(self):
        return f'label = {self.label}'


class Interval(Shape):
    """Interval is a 1D basic shape."""
    
    def __init__(self, begin, end, axis=0):
        """
        Init the Interval.
        
        begin: unit in m, var
        end: unit in m, var
        axis: ???
        """
        self.begin = begin
        self.end = end
        self.axis = axis

    def __str__(self):
        """Print Interval info."""
        res = 'Interval:'
        res += f'\nbegin = {self.begin} m'
        res += f'\nend = {self.end} m'
        res += f'\naxis = {self.axis}'
        return res

    def __contains__(self, posn):
        """
        Determind if a position is inside the Interval.
        
        posn: unit in m, var, position as input
        boundaries are not consindered as "Inside"
        """
        return self.begin < posn[self.axis] < self.end
    
    def boundary(self):
        """Return the boundary."""
        pass

class Rectangle(Shape):
    """Rectangle is a 2D basic shape."""
    
    def __init__(self, bottom_left, up_right):
        """
        Init the Rectangle.
        
        bottom_left: unit in m, (2, ) array
        up_right: unit in m, (2, ) array
        """
        self.bl = bottom_left
        self.ur = up_right

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


class Geometry:
    """Constuct the geometry."""
    
    def __init__(self, dim=2, is_cyl=False):
        """
        Init the geometry.
        
        dim: dimless, int, must be in [1, 2, 3], 1:1D; 2:2D; 3:3D
        is_cyl: bool, wether the geometry is cylidrical symmetric or not
        """
        self.dim = dim
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
        return res

    def add_shape(self, shape):
        """
        Add shape to the geometry.
        
        shape: class
        1D - shape is an instance of Interval()
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
        res = False
        posn_label = self.get_label(posn)
        return res or (posn_label == label)
