import trimesh
import numpy as np
mesh = trimesh.load('/mnt/e/Users/DELL/Desktop/stanford_data/spot.obj')
points, _ = trimesh.sample.sample_surface(mesh, 20000) 
np.savetxt('spot.xyz', points)