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
        
        posn: unit in m, var or 1D array, position as input
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
        
        bottom_left: unit in m, 2-item list, [x, y]
        up_right: unit in m, 2-item list, [x, y]
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
        
        posn: unit in m, 2-item list or 2D array, position as input
        boundaries are not consindered as "Inside"
        """
        return ((self.bl[0] < posn[0] < self.ur[0]) and 
                (self.bl[1] < posn[1] < self.ur[1]))


class Geometry:
    def __init__(self, dim=1, iscyl=False):
        self.dim = dim
        self.iscyl = iscyl
        self.sequence = list()  # diff btw list() and []

    def __str__(self):
        res = f'Geometry dimension {self.dim}D'
        if self.iscyl:
            res += ' cylindrical'
        res += '\nGeometry sequence:'
        for shape in self.sequence:
            res += '\n' + str(shape)
        return res

    def add_shape(self, shape):  # is this shape a class?
        self.sequence.append(shape)

    def get_label(self, posn):  # overlap?
        label = None
        for shape in self.sequence:
            if posn in shape:
                label = shape.label
        return label

    def find_in_label(self, posn, label):  # ??? don't understand
        res = False
        for shape in self.sequence:
            if shape.label == label:
                res = res or (posn in shape)
        return res
