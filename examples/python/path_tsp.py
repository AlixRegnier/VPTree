from vptree import VPTree
from typing import Iterable, List
from math import sqrt, pow

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def nearest_neighbor_heuristic(points : Iterable[Point]) -> List[Point]:
    #Prepare datastructure storing result
    tsp_path = [None]*len(points)

    #Define a function to compute a distance between two elements
    square_dist = lambda a, b: sqrt(pow(a.x-b.x, 2) + pow(a.y-b.y, 2))

    #Construction complexity: O(nlog(n))
    metric_tree = VPTree(points, square_dist)

    #Query parameter >= 0.0
    epsilon = 0.0

    #Start path TSP with a random element
    tail : Point = metric_tree.get_random_unvisited_element()
    tsp_path[0] = tail

    for i in range(1, len(points)):
        #nn.element (Point): indexed element pointer corresponding to found vertex
        #nn.vertex (int): nearest unvisited neighbor of given vertex
        #nn.distance (float): distance between found nearest neighbor and given vertex
        #Complexity: O(log(n)) to O(n). See README about epsilon
        nn = metric_tree.get_nearest_unvisited_neighbor(tail, epsilon)

        #Remove found neighbor from search space
        #Complexity: O(1)
        metric_tree.set_vertex_as_visited(nn.vertex)
        
        #Update tail (path TSP specific)
        tail = nn.element
        tsp_path[i] = nn.element

    return tsp_path