from vptree import VPTree
from typing import List, Iterable
from math import sqrt, pow
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def find_nearest_neighbors(points : Iterable[Point], queries : List[Point]):
    k = 5

    #Define a function to compute a distance between two elements
    square_dist = lambda a, b: sqrt(pow(a.x-b.x, 2) + pow(a.y-b.y, 2))
    
    #Construction complexity: O(nlog(n))
    metric_tree = VPTree(points, square_dist)

    #Query parameter >= 0.0
    epsilon = 0.0

    for i in range(len(queries)):
        #nn.element (Point): indexed element corresponding to found vertex
        #nn.vertex (int): nearest unvisited neighbor of given vertex
        #nn.distance (float): distance between found nearest neighbor and given vertex
        #Complexity: O(log(n)) to O(n). See README about epsilon
        nn = metric_tree.get_nearest_unvisited_neighbor(queries[i], epsilon)
        #Process
        #nn.element...
        #nn.vertex...
        #nn.distance...

        knn = metric_tree.get_k_nearest_unvisited_neighbor(queries[i], k, epsilon)
        #Process k subelements