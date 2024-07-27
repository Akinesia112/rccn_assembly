from compas.geometry import Sphere
from compas.geometry import Vector
from Example_01_reachability_map_1D import sphere, points_on_sphere_generator, robot, options


def sphere_generator():
    sphere = Sphere((0.35, 0, 0), 0.15)
    for x in range(5):
        for z in range(7):
            center = sphere.point + Vector(x, 0, z) * 0.05
            yield Sphere(center, sphere.radius)
def generator():
    for sphere in sphere_generator():
        yield points_on_sphere_generator(sphere)

'''
We can easily check the shape of the reachability as follows:

>>> map.shape       
(35, 163)           
Visualizing the results in Rhino/GH gives an indication of the best position for the tested spheres. 
The score ranges now from 436 to 937. We can ask the map for the best score, returning the score and the index:

>>> map.best_score  
(937, 20)     

Note:
Please note that the points of the point cloud correspond to 
the first frame point of each of the 2D lists and NOT the sphere center. 
This can however be changed by passing the sphere_centers for the artist to override.

>>> points, colors = artist.draw(points=sphere_centers)   
'''