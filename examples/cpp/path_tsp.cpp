#include <vptree.h>
#include <vector>

struct Point
{
    double x, y;
};

using namespace vptree;

std::vector<Point> nearest_neighbor_heuristic(const std::vector<Point>& points)
{
    //VPTree parameters
    const std::size_t n = points.size();
    double epsilon = 0.0; //A value >= 0.0

    //Prepare datastructure storing result
    std::vector<Point> tsp_path;
    tsp_path.reserve(n);

    //Optional map function (we could use directly 'points[v]' instead of 'map(v)')
    //Note that 'vertex_t' is an alias for 'std::uint64_t'
    auto map = [&](vertex_t v) -> const Point& {
        return points[v];
    };

    //Define a function to compute a distance between two vertices
    //It is up to you to provide a way to map any vertex to your element (possibly beyond n)
    auto square_dist = [&](vertex_t a, vertex_t b) -> double {
        double x2 = (map(a).x - map(b).x) * (map(a).x - map(b).x);
        double y2 = (map(a).y - map(b).y) * (map(a).y - map(b).y);

        return x2 + y2;
    };

    VPTree metric_tree(n, square_dist);

    //Start path TSP with a random Point
    vertex_t tail_vertex = metric_tree.get_random_unvisited_vertex();
    tsp_path.push_back(map(tail_vertex));

    for(std::size_t i = 1; i < n; ++i)
    {
        //nn.vertex : nearest unvisited neighbor of given vertex
        //nn.distance : distance between found nearest neighbor and given vertex
        //Complexity: O(log(n)) to O(n). See README about epsilon
        nn_t nn = metric_tree.get_nearest_unvisited_neighbor(tail_vertex, epsilon);

        //Remove found neighbor from search space
        //Complexity: O(1)
        metric_tree.set_vertex_as_visited(nn.vertex);
        
        //Update tail (path TSP specific)
        tail_vertex = nn.vertex;
        tsp_path.push_back(map(nn.vertex));
    }

    return tsp_path;
}