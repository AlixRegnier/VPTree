#ifndef VPTREE_H
#define VPTREE_H

#include <algorithm>
#include <functional>
#include <vector>
#include <stdexcept>
#include <random>
#include <cstdint>

namespace vptree
{
    using vertex_t = std::uint64_t;

    template <typename T>
    using DistanceFunction = std::function<double(const T&, const T&)>;

    template <typename T>
    struct nn_t {
        const T* element_ptr;
        vertex_t vertex;
        double distance;
    };

    template <typename T>
    class VPTree
    {

    private:
        // Internal struct used to return median partitioning result
        struct partition_result_t
        {
            std::size_t split_index;
            double median;
        };

        // Internal struct used to represent a node
        struct VPTreeNode
        {
            VPTreeNode* parent = nullptr;
            VPTreeNode* left = nullptr;
            VPTreeNode* right = nullptr;
            vertex_t pivot;

            //Median of pivot distances from other vertices
            double threshold; 

            //Whether the node and all its subtrees can be skipped
            bool skip = false;

            VPTreeNode() : VPTreeNode(nullptr, nullptr, nullptr, vertex_t{0}, 0.0, false) {}

            explicit VPTreeNode(vertex_t pivot) : VPTreeNode(nullptr, nullptr, nullptr, pivot, 0.0, false) {}

            VPTreeNode(VPTreeNode* parent, VPTreeNode* left, VPTreeNode* right, vertex_t pivot, double threshold, bool skip)
                : parent(parent), left(left), right(right), pivot(pivot), threshold(threshold), skip(skip) {}
        };

        //Nodes
        std::vector<VPTreeNode> nodes;

        //Redirection vector
        std::vector<const T*> elements;

        //Swap-and-pop vector tracking remaining vertices positions
        std::vector<std::size_t> remaining_vertices_position;

        //Swap-and-pop vector tracking remaining vertices
        std::vector<vertex_t> remaining_vertices;

        //Bitmap vector telling if a vertex is set as "visited"
        std::vector<bool> already_added_vertices;

        //Distance function used to compare two elements
        DistanceFunction<T> dist_func;

        /// @brief Recursive function for constructing the metric tree
        /// @param begin Vertex ID start iterator
        /// @param end Vertex ID end iterator
        /// @param parent Parent node address
        /// @return Initialized node address
        virtual VPTreeNode* init_node(
            std::vector<vertex_t>::iterator begin,
            std::vector<vertex_t>::iterator end,
            VPTreeNode* parent = nullptr
        );

        /// @brief Store in nn_t<T> reference, the nearest unvisited neighbor to a given query and the distance
        /// @param node Root of search
        /// @param query Element which we want to find the nearest-neighbor
        /// @param result Output containing both the nearest unvisited neighbor of 'query' and their distance
        /// @param epsilon Error factor for early search termination
        virtual void get_nearest_unvisited_neighbor(
            const VPTreeNode& node,
            const T& query,
            nn_t<T>& result,
            double epsilon = 0.0
        ) const;

        //From distances median, it partitions vertices according to median
        //Left part is for elements lesser or equal to median.
        //Returns struct { size_t split_index, double median }

        /// @brief Partition given range in two parts according to median (<=, >)
        /// @param vertices_begin Vertex ID start iterator (must not contain pivot)
        /// @param vertices_end Vertex ID end iterator (must not contain pivot)
        /// @param distances Distances computed between pivot and all vertices ID in range
        /// @return Struct containing median and the second partition starting index
        static partition_result_t partition_vertices_by_median_distance(
            std::vector<vertex_t>::iterator vertices_begin,
            std::vector<vertex_t>::iterator vertices_end,
            const std::vector<double>& distances
        );
    public:
        /// @brief Construct a VPTree using 'dist_func' on given input elements 
        /// @tparam It Iterator over elements that can be converted as T
        /// @tparam -
        /// @param begin Start iterator
        /// @param end End iterator
        /// @param dist_func Distance function for computing the distance between two T references
        template<typename It, 
             typename = std::enable_if_t<std::is_convertible_v<
                 typename std::iterator_traits<It>::value_type, T>>>
        VPTree(It begin, It end, const DistanceFunction<T>& dist_func);

        VPTree() = delete;
        VPTree(const VPTree&) = default;
        VPTree(VPTree&&) = default;
        virtual ~VPTree() = default;

        /// @brief Tell if there are no more remaining elements
        /// @return true if there are no more remaining elements, false otherwise
        virtual bool empty() const;

        /// @brief Return the element associated to given vertex
        /// @param vertex Unsigned integer in [0; N[
        /// @return A pointer to const element T associated with input vertex
        virtual const T* get_element_from_vertex(vertex_t vertex) const;

        /// @brief Return the nearest unvisited neighbor to a given query and the distance to the query
        /// @param query Element which we want to find the nearest-neighbor
        /// @param epsilon Error factor for early search termination
        /// @return A struct containing both the nearest unvisited neighbor found and the distance to the query
        virtual nn_t<T> get_nearest_unvisited_neighbor(const T& query, double epsilon = 0.0) const;

        /// @brief Return a random unvisited element
        /// @return A pointer to const element T
        virtual const T* get_random_unvisited_element() const;

        /// @brief Return a random unvisited vertex
        /// @return A vertex ID
        virtual vertex_t get_random_unvisited_vertex() const;

        /// @brief Return a vector of remaining elements
        /// @return A vector of pointers, pointing to const T elements
        virtual std::vector<const T*> get_remaining_elements_ptr() const;

        /// @brief Set given vertex as a possible candidate for a query
        /// @param vertex The vertex ID
        virtual void set_vertex_as_unvisited(vertex_t vertex);

        /// @brief Hide given vertex from search space
        /// @param vertex The vertex ID
        virtual void set_vertex_as_visited(vertex_t vertex);

        /// @brief Return the number of remaining elements
        /// @return The number of remaining elements
        virtual std::size_t remaining_size() const;

        /// @brief Return the number of elements in the VPTree
        /// @return The number of elements in the VPTree
        virtual std::size_t size() const;
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

    template <typename T>
    template <typename It, typename>
    inline VPTree<T>::VPTree(It begin, It end, const DistanceFunction<T>& dist_func)
    {
        const std::size_t size = static_cast<std::size_t>(std::distance(begin, end));

        if(size == 0)
            throw VPTreeError("VPTree", "()", "Attempted to initialize VPTree with no elements");

        nodes.reserve(size);
        elements.reserve(size);

        remaining_vertices.resize(size);
        remaining_vertices_position.resize(size);
        already_added_vertices.resize(size);

        for(std::size_t i = 0; i < size; ++i)
        {
            remaining_vertices[i] = vertex_t{i};
            remaining_vertices_position[i] = i;
        }

        for(auto it = begin; it != end; ++it)
        {
            const T& element = *it;
            elements.push_back(&element);
        }

        this->dist_func = dist_func;

        std::vector<vertex_t> vertices(remaining_vertices);
        init_node(vertices.begin(), vertices.end());
    }

    template <typename T>
    inline bool VPTree<T>::empty() const
    {
        return remaining_vertices.empty();
    }

    template <typename T>
    inline const T* VPTree<T>::get_element_from_vertex(vertex_t vertex) const
    {
        if(vertex >= elements.size())
            throw VPTreeError("VPTree", "get_element_from_vertex", "Vertex id is out of range");

        return elements[vertex];
    }

    template <typename T>
    inline const T* VPTree<T>::get_random_unvisited_element() const
    {
        if(empty())
            throw VPTreeError("VPTree", "get_random_unvisited_element", "No remaining elements");

        return elements[get_random_unvisited_vertex()];
    }

    template <typename T>
    inline vertex_t VPTree<T>::get_random_unvisited_vertex() const
    {
        if(empty())
            throw VPTreeError("VPTree", "get_random_unvisited_element", "No remaining vertices");

        std::size_t idx = static_cast<std::size_t>(RNG::rand_u64(0, remaining_vertices.size()));
        return remaining_vertices[idx];
    }

    template <typename T>
    inline std::vector<const T*> VPTree<T>::get_remaining_elements_ptr() const
    {
        std::vector<const T*> remaining_elements_ptr;
        remaining_elements_ptr.resize(remaining_vertices.size());

        for(std::size_t i = 0; i < remaining_vertices.size(); ++i)
            remaining_elements_ptr[i] = elements[remaining_vertices[i]];

        return remaining_elements_ptr;
    }

    //public definition
    template <typename T>
    inline nn_t<T> VPTree<T>::get_nearest_unvisited_neighbor(const T& query, double epsilon) const
    {
        if(empty())
            throw VPTreeError("VPTree", "get_nearest_unvisited_neighbor", "No remaining neighbors");

        if(epsilon < 0.0)
            throw VPTreeError("VPTree", "get_nearest_unvisited_neighbor", "epsilon must be null or positive");

        nn_t<T> result = {
            elements[remaining_vertices[0]],
            remaining_vertices[0],
            dist_func(query, *elements[remaining_vertices[0]])
        };

        get_nearest_unvisited_neighbor(nodes[0], query, result, epsilon);

        return result;
    }

    //private definition
    template <typename T>
    void VPTree<T>::get_nearest_unvisited_neighbor(const VPTreeNode& node, const T& query, nn_t<T>& result, double epsilon) const
    {
        double distance = dist_func(*elements[node.pivot], query);

        if(!already_added_vertices[node.pivot] && distance < result.distance)
        {
            result.element_ptr = elements[node.pivot];
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

    template <typename T>
    inline typename VPTree<T>::VPTreeNode* VPTree<T>::init_node(std::vector<vertex_t>::iterator begin, std::vector<vertex_t>::iterator end, VPTreeNode* parent)
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
        VPTreeNode& node = nodes.back();
        node.parent = parent;

        {
            //Select a random vertex as pivot
            std::size_t pivot_index = static_cast<std::size_t>(RNG::rand_u64(0, size));

            node.pivot = begin[pivot_index];

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
                distances[i] = dist_func(*elements[node.pivot], *elements[begin[i]]);

            partition_result_t pr = partition_vertices_by_median_distance(begin, end-1, distances);

            split_index = pr.split_index;
            node.threshold = pr.median;
        }

        node.left = init_node(begin, begin+split_index, &node);
        node.right = init_node(begin+split_index, end-1, &node);

        return &node;
    }

    template <typename T>
    typename VPTree<T>::partition_result_t VPTree<T>::partition_vertices_by_median_distance(std::vector<vertex_t>::iterator vertices_begin, std::vector<vertex_t>::iterator vertices_end, const std::vector<double>& distances)
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


    template <typename T>
    inline std::size_t VPTree<T>::remaining_size() const
    {
        return remaining_vertices.size();
    }

    template <typename T>
    inline void VPTree<T>::set_vertex_as_unvisited(vertex_t vertex)
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

    template <typename T>
    inline void VPTree<T>::set_vertex_as_visited(vertex_t vertex)
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

    template <typename T>
    inline std::size_t VPTree<T>::size() const
    {
        return nodes.size();
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