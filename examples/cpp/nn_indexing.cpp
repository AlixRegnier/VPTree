#include <vptree.hpp>
#include <vector>
#include <cmath>

struct Point
{
    double x, y;
};

using namespace vptree;

void find_nearest_neighbors(const std::vector<Point> points, const std::vector<Point> queries)
{
    const std::size_t k = 5;

    //Define a function to compute a distance between two elements
    auto square_dist = [&](const Point& a, const Point& b) -> double {
        double x2 = (a.x - b.x) * (a.x - b.x);
        double y2 = (a.y - b.y) * (a.y - b.y);
        
        return sqrt(x2 + y2);
    };
    
    //Construction complexity: O(nlog(n))
    VPTree<Point> metric_tree(points.begin(), points.end(), square_dist);
    
    //Query parameter >= 0.0
    double epsilon = 0.0;

    for(std::size_t i = 0; i < queries.size(); ++i)
    {        
        //nn.element_ptr (const Point*): indexed element pointer corresponding to found vertex
        //nn.vertex : nearest unvisited neighbor of given vertex
        //nn.distance : distance between found nearest neighbor and given vertex
        //Complexity: O(log(n)) to O(n). See README about epsilon
        nn_t<Point> nn = metric_tree.get_nearest_unvisited_neighbor(queries[i], epsilon);
        //Process 
        //nn.element_ptr...
        //nn.vertex...
        //nn.distance...

        //Complexity: O(k.log(n)) to O(kn). See README about epsilon
        std::vector<nn_t<Point>> knn = metric_tree.get_k_nearest_unvisited_neighbors(queries[i], k, epsilon);
        //Process k subelements (ascending sort by distance then by vertex ID)
        //...
    }
}