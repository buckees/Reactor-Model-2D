"""
This is the Geometry module.

Geometry module defines the basic shapes, such as 
1D: Interval 
2D: Rectangle, Triangle, etc.

Geometry module defines the construction of the geometry.
"""


class Shape:  # Shape does not sound like a top level class
    def __init__(self, label):
        self.label = label

    def __str__(self):
        return f'label = {self.label}'


class Interval(Shape):
    def __init__(self, label, begin, end, axis=0):
        # since label is already defined in Shape, is label here duplicate?
        super().__init__(label)  # what does super mean? super() vs. no super()
        self.begin = begin
        self.end = end
        self.axis = axis

    def __str__(self):
        res = 'Interval:'
        res += '\n' + super().__str__()
        res += f'\nbegin = {self.begin}'
        res += f'\nend = {self.end}'
        res += f'\naxis = {self.axis}'
        return res

    def __contains__(self, location):
        return self.begin < location[self.axis] < self.end


class Rectangle(Shape):
    pass


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

    def get_label(self, location):  # overlap?
        label = None
        for shape in self.sequence:
            if location in shape:
                label = shape.label
        return label

    def find_in_label(self, location, label):  # ??? don't understand
        res = False
        for shape in self.sequence:
            if shape.label == label:
                res = res or (location in shape)
        return res
