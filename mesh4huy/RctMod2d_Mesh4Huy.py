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

import numpy as np


"""Import plasma modules."""
from RctMod2d_Geom import Geom2d, Domain, Rectangle
from RctMod2d_Mesh import Mesh2d

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



geom2d.plot(fname='geom2d.png', figsize=(5, 8), ihoriz=0)
print(geom2d)
# generate mesh to imported geometry
mesh2d = Mesh2d()
mesh2d.import_geom(geom2d)
mesh2d.generate_mesh(ngrid=(51, 41))
mesh2d.plot(fname='mesh2d.png', figsize=(5, 8), ihoriz=0)

# save matrix in numpy binary format
np.save('mesh_x', mesh2d.x)
np.save('mesh_z', mesh2d.z)
np.save('mesh_mat', mesh2d.mat)

# save matrix in csv format
np.savetxt("mesh_x.csv", mesh2d.x, delimiter=",")
np.savetxt("mesh_z.csv", mesh2d.z, delimiter=",")
np.savetxt("mesh_mat.csv", mesh2d.mat, delimiter=",")