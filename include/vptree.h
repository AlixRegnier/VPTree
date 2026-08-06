#ifndef VPTREE_H
#define VPTREE_H

#include <algorithm> //Sort algorithms
#include <functional> //Function prototyping
#include <vector>
#include <stdexcept>
#include <random>
#include <cstdint>

namespace vptree
{
    using vertex_t = std::uint64_t;
    using DistanceFunction = std::function<double(vertex_t, vertex_t)>;

    struct nn_t {
        vertex_t vertex;
        double distance;
    };

    class VPTree
    {

    private:
        struct VPTreeNode
        {
            VPTreeNode* left = nullptr;
            VPTreeNode* right = nullptr;
            VPTreeNode* parent = nullptr;
            vertex_t pivot;

            double threshold; //Median of pivot distances from other vertices
            bool skip = false;

            VPTreeNode() : VPTreeNode(nullptr, nullptr, nullptr, vertex_t{0}, double{0}, bool{0}) {}

            VPTreeNode(vertex_t pivot) : VPTreeNode(nullptr, nullptr, nullptr, pivot, double{0}, bool{0}) {}

            VPTreeNode(VPTreeNode* parent, VPTreeNode* left, VPTreeNode* right, vertex_t pivot, double threshold, bool skip)
                : parent(parent), left(left), right(right), pivot(pivot), threshold(threshold), skip(skip) {}
        };

        //Internal nodes
        std::vector<VPTreeNode> nodes;

        //Swap-and-pop utility
        std::vector<std::size_t> remaining_vertices_position;
        std::vector<vertex_t> remaining_vertices;
        std::vector<bool> already_added_vertices;

        DistanceFunction dist_func;

        struct partition_result_t
        {
            std::size_t split_index;
            double median;
        };

        void get_nearest_unvisited_neighbor(
            const VPTreeNode& node,
            vertex_t query, nn_t& result,
            double epsilon = 0.0
        ) const;

        VPTreeNode* init_node(
            std::vector<vertex_t>::iterator begin,
            std::vector<vertex_t>::iterator end,
            VPTreeNode* parent = nullptr
        );

        //From distances median, it partitions vertices according to median
        //Left part is for elements lesser or equal to median.
        //Returns struct { size_t split_index, double median }
        static partition_result_t partition_vertices_by_median_distance(
            const std::vector<vertex_t>::iterator vertices_begin,
            const std::vector<vertex_t>::iterator vertices_end,
            const std::vector<double>& distances
        );
    public:
        VPTree(std::size_t n, const DistanceFunction& dist_func);
        VPTree() = delete;
        VPTree(const VPTree&) = delete;
        VPTree(VPTree&&) = delete;
        ~VPTree() = default;

        void set_vertex_as_visited(vertex_t vertex);
        void set_vertex_as_unvisited(vertex_t vertex);

        nn_t get_nearest_unvisited_neighbor(vertex_t query, double epsilon = 0.0) const;
        vertex_t get_random_unvisited_vertex() const;
    };

    class RNG
    {
        private:
            static std::mt19937 gen;
            static std::uint64_t seed;
            RNG() = delete;
        public:
            static std::uint64_t rand_u64();
            static std::uint64_t rand_u64(std::uint64_t a, std::uint64_t b);
            static void set_seed(std::uint64_t seed);
            static std::uint64_t get_seed();
            static std::uint64_t get_random_seed();
    };

    class VPTreeError : public std::runtime_error
    {
    public:
        VPTreeError(const std::string& class_name, const std::string& function_name, const std::string& msg)
            : std::runtime_error("[ERROR] vptree::" + class_name + "::" + function_name + " : " + msg) {}
    };

    //VPTree implementation
    VPTree::VPTree(std::size_t n, const DistanceFunction& dist_func)
    {
        if(n == 0)
            throw VPTreeError("VPTree", "()", "Attempted to initialize VPTree with no elements");

        this->nodes.reserve(n);

        this->remaining_vertices.resize(n);
        this->remaining_vertices_position.resize(n);
        this->already_added_vertices.resize(n);

        for(std::size_t i = 0; i < n; ++i)
        {
            this->remaining_vertices[i] = vertex_t{i};
            this->remaining_vertices_position[i] = i;
        }

        this->dist_func = dist_func;

        std::vector<vertex_t> vertices(remaining_vertices);
        init_node(vertices.begin(), vertices.end());
    }

    VPTree::VPTreeNode* VPTree::init_node(std::vector<vertex_t>::iterator begin, std::vector<vertex_t>::iterator end, VPTreeNode* parent)
    {
        if(begin >= end)
            return nullptr;

        const std::size_t size = static_cast<std::size_t>(std::distance(begin, end));

        if(size == 1)
        {
            nodes.emplace_back(begin[0]);
            return &nodes.back();
        }

        nodes.emplace_back();
        VPTreeNode* node = &nodes.back();
        node->parent = parent;

        {
            //Select a random vertex as pivot
            std::size_t pivot_index = static_cast<std::size_t>(RNG::rand_u64(0, size));

            node->pivot = begin[pivot_index];

            //Swap pivot with last element of range
            std::swap(begin[pivot_index], begin[size-1]);
        }

        std::size_t split_index;

        //Partition left and right according to median
        {
            std::vector<double> distances;
            distances.resize(size-1);

            //Compute distances
            for(std::size_t i = 0; i+1 < size; ++i)
                distances[i] = dist_func(node->pivot, begin[i]);

            partition_result_t pr = partition_vertices_by_median_distance(begin, end-1, distances);

            split_index = pr.split_index;
            node->threshold = pr.median;
        }

        node->left = init_node(begin, begin+split_index, node);
        node->right = init_node(begin+split_index, end-1, node);

        return node;
    }

    VPTree::partition_result_t VPTree::partition_vertices_by_median_distance(std::vector<vertex_t>::iterator vertices_begin, std::vector<vertex_t>::iterator vertices_end, const std::vector<double>& distances)
    {
        const std::size_t size = static_cast<std::size_t>(std::distance(vertices_begin, vertices_end));

        if (size == 0)
            throw VPTreeError("VPTree", "partition_vertices_by_median_distance", "got empty range");

        //Zip into pairs so both vectors move together.
        std::vector<std::pair<double, vertex_t>> pairs;
        pairs.reserve(size);
        for (std::size_t i = 0; i < size; ++i)
            pairs.emplace_back(distances[i], std::move(vertices_begin[i]));

        auto byFirst = [](const auto& a, const auto& b) { return a.first < b.first; };

        //Introselect
        auto mid = pairs.begin() + size / 2;
        std::nth_element(pairs.begin(), mid, pairs.end(), byFirst);

        double median;
        if (size % 2 == 1) {
            median = mid->first; // odd size: exact middle element
        } else {
            // even size: average of the two middle elements.
            // mid->first is the upper median
            // The lower median is the max of everything before mid.
            double upper = mid->first;
            double lower = std::max_element(pairs.begin(), mid, byFirst)->first;
            median = (lower + upper) / 2.0;
        }

        auto splitPoint = std::partition(
            pairs.begin(), pairs.end(),
            [median](const auto& p) { return p.first <= median; }
        );
        std::size_t split_index = static_cast<std::size_t>(std::distance(pairs.begin(), splitPoint));

        //Write back vertices
        for (std::size_t i = 0; i < size; ++i)
            vertices_begin[i] = std::move(pairs[i].second);

        return { split_index, median };
    }

    nn_t VPTree::get_nearest_unvisited_neighbor(vertex_t query, double epsilon) const
    {
        if(remaining_vertices.empty())
            throw VPTreeError("VPTree", "get_nearest_unvisited_neighbor", "No remaining neighbors");

        if(epsilon < 0.0)
            throw VPTreeError("VPTree", "get_nearest_unvisited_neighbor", "epsilon must be null or positive");

        nn_t result = {
            remaining_vertices[0],
            dist_func(query, remaining_vertices[0])
        };

        get_nearest_unvisited_neighbor(nodes[0], query, result, epsilon);

        return result;
    }

    void VPTree::get_nearest_unvisited_neighbor(const VPTreeNode& node, vertex_t query, nn_t& result, double epsilon) const
    {
        double distance = dist_func(node.pivot, query);

        if(!already_added_vertices[node.pivot] && distance < result.distance)
        {
            result.vertex = node.pivot;
            result.distance = distance;
        }

        double relaxed_tau = result.distance / (1.0 + epsilon);

        if(distance < node.threshold)
        {
            if(node.left != nullptr && !node.left->skip && (distance - relaxed_tau) <= node.threshold)
                get_nearest_unvisited_neighbor(*node.left, query, result, epsilon);

            relaxed_tau = result.distance / (1.0 + epsilon);

            if(node.right != nullptr && !node.right->skip && (distance + relaxed_tau) >= node.threshold)
                get_nearest_unvisited_neighbor(*node.right, query, result, epsilon);
        }
        else
        {
            if(node.right != nullptr && !node.right->skip && (distance + relaxed_tau) >= node.threshold)
                get_nearest_unvisited_neighbor(*node.right, query, result, epsilon);

            relaxed_tau = result.distance / (1.0 + epsilon);

            if(node.left != nullptr && !node.left->skip && (distance - relaxed_tau) <= node.threshold)
                get_nearest_unvisited_neighbor(*node.left, query, result, epsilon);
        }
    }

    void VPTree::set_vertex_as_visited(vertex_t vertex)
    {
        if(vertex >= nodes.size())
            throw VPTreeError("VPTree", "set_vertex_as_visited", "Vertex id is out of range");

        if(already_added_vertices[vertex])
            return;

        //Set node as visited
        already_added_vertices[vertex] = true;

        //Update remaining nodes (swap-and-pop)
        remaining_vertices[remaining_vertices_position[vertex]] = remaining_vertices.back();
        remaining_vertices_position[remaining_vertices.back()] = remaining_vertices_position[vertex];
        remaining_vertices.pop_back();

        VPTreeNode* node = &nodes[vertex];
        while(node != nullptr)
        {
            node->skip = (node->left == nullptr || node->left->skip) && (node->right == nullptr || node->right->skip) && already_added_vertices[node->pivot];

            //Stop property propagation if not masked
            if(!node->skip)
                return;

            node = node->parent;
        }
    }

    void VPTree::set_vertex_as_unvisited(vertex_t vertex)
    {
        if(vertex >= nodes.size())
            throw VPTreeError("VPTree", "set_vertex_as_unvisited", "Vertex id is out of range");

        if(!already_added_vertices[vertex])
            return;

        //Set node as unvisited
        already_added_vertices[vertex] = false;

        //Update remaining nodes (swap-and-pop)
        remaining_vertices.push_back(vertex);
        remaining_vertices_position[vertex] = remaining_vertices.size() - 1;

        VPTreeNode* node = &nodes[vertex];
        while(node != nullptr)
        {
            //Stop property propagation if not masked
            if(!node->skip)
                return;

            node->skip = false;

            node = node->parent;
        }
    }

    vertex_t VPTree::get_random_unvisited_vertex() const
    {
        if(remaining_vertices.empty())
            throw VPTreeError("VPTree", "get_random_unvisited_vertex", "No remaining vertices");

        std::size_t idx = static_cast<std::size_t>(RNG::rand_u64(0, remaining_vertices.size()));
        return remaining_vertices[idx];
    }

    // RNG implementation
    std::uint64_t RNG::rand_u64()
    {
        return gen();
    }

    std::uint64_t RNG::rand_u64(std::uint64_t a, std::uint64_t b) //Generate an unsigned integer in [a ; b[
    {
        if(a >= b)
            throw VPTreeError("RNG", "rand_u64", "got unexpected interval");

        return (rand_u64() % (b-a)) + a;
    }

    std::uint64_t RNG::get_seed()
    {
        return seed;
    }

    void RNG::set_seed(std::uint64_t new_seed)
    {
        seed = new_seed;
        gen.seed(seed);
    }

    std::uint64_t RNG::get_random_seed()
    {
        return std::random_device()();
    }

    std::uint64_t RNG::seed = std::uint64_t{42};
    std::mt19937 RNG::gen = std::mt19937(RNG::seed); // Standard mersenne_twister_engine seeded with default random_device
};

#endif