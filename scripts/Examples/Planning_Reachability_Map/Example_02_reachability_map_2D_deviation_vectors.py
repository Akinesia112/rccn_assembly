import math
from compas.geometry import Frame
from compas_fab.robots import DeviationVectorsGenerator
from compas_fab.robots import ReachabilityMap
from Example_01_reachability_map_1D import sphere, points_on_sphere_generator, robot, options

def deviation_vector_generator(frame):
    for xaxis in DeviationVectorsGenerator(frame.xaxis, math.radians(40), 1):
        yaxis = frame.zaxis.cross(xaxis)
        yield Frame(frame.point, xaxis, yaxis)

# And combine both generators into a generator that yields a list of a list of frames (2D).

def generator():
    for frame in points_on_sphere_generator(sphere):
        yield deviation_vector_generator(frame)

'''
The robot is set up as in Example 01 and we calculate the reachability map as follows.

>>> map = ReachabilityMap()                                                      
>>> map.calculate(generator(), robot, options)                                   
>>> map.to_json(os.path.join(DATA, "reachability", "map2D_deviation.json")) 

The visualization of the results in Rhino/GH looks different than in Example 01, 
as the scores per evaluated point are now more scattered and range between 16 (dark purple) and 26 (yellow).
'''  