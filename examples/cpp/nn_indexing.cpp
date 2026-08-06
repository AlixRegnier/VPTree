#include <vptree.h>
#include <vector>

struct Point
{
    double x, y;
};

using namespace vptree;

void index(std::vector<Point> points, const std::vector<Point> queries)
{
    //VPTree parameter
    const std::size_t n = points.size();
    double epsilon = 0.0; //A value >= 0.0

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

    //Construction complexity: O(nlog(n))
    VPTree metric_tree(n, square_dist);

    //Make little room for incoming queries
    points.reserve(points.size()+1);
    const vertex_t query_vertex = static_cast<vertex_t>(points.size()-1);

    for(std::size_t i = 0; i < queries.size(); ++i)
    {
        //Fetch query
        points.back() = queries[i]; 
        
        //Find nearest neighbor
        //nn.vertex : nearest unvisited neighbor of given vertex
        //nn.distance : distance between found nearest neighbor and given vertex
        //Complexity: O(log(n)) to O(n). See README about epsilon
        nn_t nn = metric_tree.get_nearest_unvisited_neighbor(query_vertex, epsilon);

        //Process 
        //nn.vertex...
    }
}