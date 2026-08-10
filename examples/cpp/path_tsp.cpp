#include <vptree.h>
#include <vector>

struct Point
{
    double x, y;
};

using namespace vptree;

std::vector<Point> nearest_neighbor_heuristic(const std::vector<Point>& points)
{
    //Prepare datastructure storing result
    std::vector<Point> tsp_path;
    tsp_path.reserve(n);

    //Define a function to compute a distance between two elements
    auto square_dist = [&](const Point& a, const Point& b) -> double {
        double x2 = (a.x - b.x) * (a.x - b.x);
        double y2 = (a.y - b.y) * (a.y - b.y);

        return x2 + y2;
    };

    //Construction complexity: O(nlog(n))
    VPTree<Point> metric_tree(points.begin(), points.end(), square_dist);

    //Query parameter >= 0.0
    double epsilon = 0.0;

    //Start path TSP with a random element
    const Point* tail = metric_tree.get_random_unvisited_element();
    tsp_path.push_back(*tail);

    for(std::size_t i = 1; i < n; ++i)
    {
        //nn.element_ptr (const Point*): indexed element pointer corresponding to found vertex
        //nn.vertex (vertex_t): nearest unvisited neighbor of given vertex
        //nn.distance (double): distance between found nearest neighbor and given vertex
        //Complexity: O(log(n)) to O(n). See README about epsilon
        nn_t<Point> nn = metric_tree.get_nearest_unvisited_neighbor(*tail, epsilon);

        //Remove found neighbor from search space
        //Complexity: O(1)
        metric_tree.set_vertex_as_visited(nn.vertex);
        
        //Update tail (path TSP specific)
        tail = nn.element_ptr;
        tsp_path.push_back(*nn.element_ptr);
    }

    return tsp_path;
}