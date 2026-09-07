#ifndef VPTREE_H
#define VPTREE_H

#include <algorithm>
#include <functional>
#include <vector>
#include <stdexcept>
#include <random>
#include <cstdint>
#include <cstddef>

namespace vptree
{
    using vertex_t = std::uint64_t;

    template <typename T, typename dist_t = double>
    using DistanceFunction = std::function<dist_t(const T&, const T&)>;

    template <typename T, typename dist_t = double>
    struct nn_t {
        const T* element_ptr;
        vertex_t vertex;
        dist_t distance;

        nn_t(const T* element_ptr, vertex_t vertex, dist_t distance) : element_ptr(element_ptr), vertex(vertex), distance(distance) {};
    };

    //Note: for smaller K, sorted list is better when need for deleting duplicates. Do not use it for k > ~15.
    template <typename T, typename dist_t = double>
    class TopKNeighbors 
    {
    private:
        std::size_t k;
        std::vector<nn_t<T, dist_t>> data;

        static bool lt(const nn_t<T, dist_t>& a, const nn_t<T, dist_t>& b)
        {
            if(a.distance != b.distance)
                return a.distance < b.distance;

            return a.vertex > b.vertex; //make deep vertices more likely to be selected
        }

    public:
        explicit TopKNeighbors(std::size_t k) : k(k)
        {
            data.reserve(k+1);
        }

        void push(const nn_t<T, dist_t>& value)
        {
            auto pos = std::lower_bound(data.begin(), data.end(), value, lt);

            // Same ID => duplicate.
            if (pos != data.end() && pos->vertex == value.vertex)
                return;

            data.insert(pos, value);

            if(data.size() > k)
                data.pop_back();
        }

        dist_t farthest_distance() const
        {
            return data.back().distance;
        }

        const std::vector<nn_t<T, dist_t>>& vec() const noexcept
        {
            return data;
        }

        std::vector<nn_t<T, dist_t>>& vec() noexcept
        {
            return data;
        }

        std::size_t size() const noexcept
        {
            return data.size();
        }

        std::size_t max_size() const noexcept
        {
            return k;
        }
    };
        

    template <typename T, typename dist_t = double>
    class VPTree
    {

    private:
        // Internal struct used to return median partitioning result
        struct partition_result_t
        {
            std::size_t split_index;
            dist_t median;
        };

        // Internal struct used to represent a node
        struct VPTreeNode
        {
            VPTreeNode* parent = nullptr;
            VPTreeNode* left = nullptr;
            VPTreeNode* right = nullptr;
            vertex_t pivot;

            //Median of pivot distances from other vertices
            dist_t threshold; 

            //Whether the node and all its subtrees can be skipped
            bool skip = false;

            explicit VPTreeNode() : VPTreeNode(nullptr, nullptr, nullptr, vertex_t{0}, dist_t{0}, false) {}

            explicit VPTreeNode(vertex_t pivot) : VPTreeNode(nullptr, nullptr, nullptr, pivot, dist_t{0}, false) {}

            explicit VPTreeNode(VPTreeNode* parent, VPTreeNode* left, VPTreeNode* right, vertex_t pivot, dist_t threshold, bool skip)
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
        DistanceFunction<T, dist_t> dist_func;

        /// @brief Return the k nearest unvisited neighbor to a given query and the distance to the query
        /// @param query Element which we want to find the k nearest unvisited neighbors
        /// @param epsilon Error factor for early search termination
        /// @return A struct containing both the k nearest unvisited neighbor found and the distance to the query
        virtual void get_k_nearest_unvisited_neighbor(
            const VPTreeNode& node,
            const T& query,
            TopKNeighbors<T, dist_t>& top_k,
            dist_t epsilon = dist_t{0}
        ) const;

        /// @brief Store in nn_t<T, dist_t> reference, the nearest unvisited neighbor to a given query and the distance
        /// @param node Root of search
        /// @param query Element which we want to find the nearest-neighbor
        /// @param result Output containing both the nearest unvisited neighbor of 'query' and their distance
        /// @param epsilon Error factor for early search termination
        virtual void get_nearest_unvisited_neighbor(
            const VPTreeNode& node,
            const T& query,
            nn_t<T, dist_t>& result,
            dist_t epsilon = dist_t{0}
        ) const;

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

        /// @brief Partition given range in two parts according to median (<=, >)
        /// @param vertices_begin Vertex ID start iterator (must not contain pivot)
        /// @param vertices_end Vertex ID end iterator (must not contain pivot)
        /// @param distances Distances computed between pivot and all vertices ID in range
        /// @return Struct containing median and the second partition starting index
        static partition_result_t partition_vertices_by_median_distance(
            std::vector<vertex_t>::iterator vertices_begin,
            std::vector<vertex_t>::iterator vertices_end,
            const std::vector<dist_t>& distances
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
        VPTree(It begin, It end, const DistanceFunction<T, dist_t>& dist_func);

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

        /// @brief Return of k random unvisited elements
        /// @param k The size of the vector
        /// @return A vector containing k random unvisited element const pointers
        virtual std::vector<const T*> get_k_random_unvisited_elements(std::size_t k) const;

        /// @brief Return the k nearest unvisited neighbors to a given query and the distance to the query
        /// @param query Element which we want to find the k nearest unvisited neighbors
        /// @param epsilon Error factor for early search termination
        /// @return A vector of structs containing both the k nearest unvisited neighbors found and the distance to the query
        virtual std::vector<nn_t<T, dist_t>> get_k_nearest_unvisited_neighbors(const T& query, std::size_t k, dist_t epsilon = dist_t{0}) const;

        /// @brief Return of k random unvisited vertices
        /// @param k The number of random unvisited elements to get
        /// @return A vector containing k random unvisited vertices
        virtual std::vector<vertex_t> get_k_random_unvisited_vertices(std::size_t k) const;

        /// @brief Return the nearest unvisited neighbor to a given query and the distance to the query
        /// @param query Element which we want to find the nearest-neighbor
        /// @param epsilon Error factor for early search termination
        /// @return A struct containing both the nearest unvisited neighbor found and the distance to the query
        virtual nn_t<T, dist_t> get_nearest_unvisited_neighbor(const T& query, dist_t epsilon = dist_t{0}) const;

        /// @brief Return a random unvisited element
        /// @return A pointer to const element T
        virtual const T* get_random_unvisited_element() const;

        /// @brief Return a random unvisited vertex
        /// @return A vertex ID
        virtual vertex_t get_random_unvisited_vertex() const;

        /// @brief Return a vector of remaining elements
        /// @return A vector of pointers, pointing to const T elements
        virtual std::vector<const T*> get_remaining_elements() const;

        /// @brief Set all vertices as a possible candidates for a query
        virtual void set_all_vertices_as_unvisited();
        
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

    struct RNG
    {
        RNG() = delete;
        static std::uint64_t seed;
        static std::mt19937 gen;
        static std::uint64_t get_random_seed();
        static std::uint64_t get_seed();
        static std::uint64_t rand_u64();
        static std::uint64_t rand_u64(std::uint64_t a, std::uint64_t b);
        static void set_seed(std::uint64_t seed);
    };

    class VPTreeError : public std::runtime_error
    {
    public:
        VPTreeError(const std::string& class_name, const std::string& function_name, const std::string& msg)
            : std::runtime_error("[ERROR] vptree::" + class_name + "::" + function_name + " : " + msg) {}
    };

    template <typename T, typename dist_t>
    template <typename It, typename>
    inline VPTree<T, dist_t>::VPTree(It begin, It end, const DistanceFunction<T, dist_t>& dist_func)
    {
        const std::size_t size = static_cast<std::size_t>(std::distance(begin, end));

        if(size == 0)
            throw VPTreeError("VPTree", "()", "Attempted to initialize VPTree with no elements");

        nodes.reserve(size);

        remaining_vertices.resize(size);
        remaining_vertices_position.resize(size);
        already_added_vertices.resize(size);
        elements.reserve(size);
        
        for(auto it = begin; it != end; ++it)
        {
            const T& element = *it;
            elements.push_back(&element);
        }

        this->dist_func = dist_func;

        for(std::size_t i = 0; i < size; ++i)
            remaining_vertices[i] = vertex_t{i};

        //After this call: remaining_vertices will contain partitioned points
        init_node(remaining_vertices.begin(), remaining_vertices.end());

        //Track new position of elements
        for(std::size_t i = 0; i < size; ++i)
            remaining_vertices_position[remaining_vertices[i]] = i;
    }

    template <typename T, typename dist_t>
    inline bool VPTree<T, dist_t>::empty() const
    {
        return remaining_vertices.empty();
    }

    template <typename T, typename dist_t>
    inline const T* VPTree<T, dist_t>::get_element_from_vertex(vertex_t vertex) const
    {
        if(vertex >= elements.size())
            throw VPTreeError("VPTree", "get_element_from_vertex", "Vertex id is out of range");

        return elements[vertex];
    }

    //public definition
    template <typename T, typename dist_t>
    inline std::vector<nn_t<T, dist_t>> VPTree<T, dist_t>::get_k_nearest_unvisited_neighbors(const T& query, std::size_t k, dist_t epsilon) const
    {
        if(remaining_size() < k)
            throw VPTreeError("VPTree", "get_k_nearest_unvisited_neighbor", "No enough remaining neighbors");

        if(epsilon < dist_t{0})
            throw VPTreeError("VPTree", "get_k_nearest_unvisited_neighbor", "epsilon must be null or positive");

        if(k == 0)
            throw VPTreeError("VPTree", "get_k_nearest_unvisited_neighbor", "k must greater or equal to 1");

        //Initial solution (no random)
        TopKNeighbors<T, dist_t> top_k(k);

        for(vertex_t v : get_k_random_unvisited_vertices(k))
        {
            top_k.push({
                elements[v],
                v,
                dist_func(query, *elements[v])
            });
        }

        get_k_nearest_unvisited_neighbor(nodes[0], query, top_k, epsilon);

        return top_k.vec();
    }

    //private definition
    template <typename T, typename dist_t>
    inline void VPTree<T, dist_t>::get_k_nearest_unvisited_neighbor(const VPTreeNode& node, const T& query, TopKNeighbors<T, dist_t>& top_k, dist_t epsilon) const
    {
        dist_t distance = dist_func(*elements[node.pivot], query);

        if(!already_added_vertices[node.pivot] && distance < top_k.farthest_distance())
        {
            top_k.push({
                elements[node.pivot], 
                node.pivot, 
                distance
            });
        }

        dist_t relaxed_tau = top_k.farthest_distance() / (dist_t{1} + epsilon);

        if(distance < node.threshold)
        {
            if(node.left != nullptr && !node.left->skip && (distance - relaxed_tau) <= node.threshold)
                get_k_nearest_unvisited_neighbor(*node.left, query, top_k, epsilon);

            relaxed_tau = top_k.farthest_distance() / (dist_t{1} + epsilon);

            if(node.right != nullptr && !node.right->skip && (distance + relaxed_tau) >= node.threshold)
                get_k_nearest_unvisited_neighbor(*node.right, query, top_k, epsilon);
        }
        else
        {
            if(node.right != nullptr && !node.right->skip && (distance + relaxed_tau) >= node.threshold)
                get_k_nearest_unvisited_neighbor(*node.right, query, top_k, epsilon);

            relaxed_tau = top_k.farthest_distance() / (dist_t{1} + epsilon);

            if(node.left != nullptr && !node.left->skip && (distance - relaxed_tau) <= node.threshold)
                get_k_nearest_unvisited_neighbor(*node.left, query, top_k, epsilon);
        }
    }

    template <typename T, typename dist_t>
    inline std::vector<const T*> VPTree<T, dist_t>::get_k_random_unvisited_elements(std::size_t k) const
    {
        if(remaining_size() < k)
            throw VPTreeError("VPTree", "get_k_random_unvisited_elements", "No enough remaining elements");

        std::vector<const T*> remaining_elements;
        remaining_elements.reserve(k);

        for(vertex_t v : get_k_random_unvisited_vertices(k))
            remaining_elements.push_back(elements[v]);

        return remaining_elements;
    }

    template <typename T, typename dist_t>
    inline std::vector<vertex_t> VPTree<T, dist_t>::get_k_random_unvisited_vertices(std::size_t k) const
    {
        if(remaining_size() < k)
            throw VPTreeError("VPTree", "get_k_random_unvisited_vertices", "No enough remaining vertices");

        std::vector<vertex_t> sampled_vertices;
        sampled_vertices.reserve(k);

        //Choose k random elements by randomly choosing one element from k partitions of same size
        //Note: remaining_vertices was a bit partitioned during construction phase
        for(std::size_t i = 0; i < k; ++i)
        {
            std::size_t j = static_cast<std::size_t>(RNG::rand_u64(i*remaining_size()/k, (i+1)*remaining_size()/k));
            sampled_vertices.push_back(remaining_vertices[j]);
        }

        return sampled_vertices;
    }

    //public definition
    template <typename T, typename dist_t>
    inline nn_t<T, dist_t> VPTree<T, dist_t>::get_nearest_unvisited_neighbor(const T& query, dist_t epsilon) const
    {
        if(empty())
            throw VPTreeError("VPTree", "get_nearest_unvisited_neighbor", "No remaining neighbors");

        if(epsilon < dist_t{0})
            throw VPTreeError("VPTree", "get_nearest_unvisited_neighbor", "epsilon must be null or positive");

        vertex_t v = get_random_unvisited_vertex();

        nn_t<T, dist_t> result = {
            elements[v],
            v,
            dist_func(query, *elements[v])
        };

        get_nearest_unvisited_neighbor(nodes[0], query, result, epsilon);

        return result;
    }

    //private definition
    template <typename T, typename dist_t>
    inline void VPTree<T, dist_t>::get_nearest_unvisited_neighbor(const VPTreeNode& node, const T& query, nn_t<T, dist_t>& result, dist_t epsilon) const
    {
        dist_t distance = dist_func(*elements[node.pivot], query);

        if(!already_added_vertices[node.pivot] && distance < result.distance)
        {
            result.element_ptr = elements[node.pivot];
            result.vertex = node.pivot;
            result.distance = distance;
        }

        dist_t relaxed_tau = result.distance / (dist_t{1} + epsilon);

        if(distance < node.threshold)
        {
            if(node.left != nullptr && !node.left->skip && (distance - relaxed_tau) <= node.threshold)
                get_nearest_unvisited_neighbor(*node.left, query, result, epsilon);

            relaxed_tau = result.distance / (dist_t{1} + epsilon);

            if(node.right != nullptr && !node.right->skip && (distance + relaxed_tau) >= node.threshold)
                get_nearest_unvisited_neighbor(*node.right, query, result, epsilon);
        }
        else
        {
            if(node.right != nullptr && !node.right->skip && (distance + relaxed_tau) >= node.threshold)
                get_nearest_unvisited_neighbor(*node.right, query, result, epsilon);

            relaxed_tau = result.distance / (dist_t{1} + epsilon);

            if(node.left != nullptr && !node.left->skip && (distance - relaxed_tau) <= node.threshold)
                get_nearest_unvisited_neighbor(*node.left, query, result, epsilon);
        }
    }

    template <typename T, typename dist_t>
    inline const T* VPTree<T, dist_t>::get_random_unvisited_element() const
    {
        if(empty())
            throw VPTreeError("VPTree", "get_random_unvisited_element", "No remaining elements");

        return elements[get_random_unvisited_vertex()];
    }

    template <typename T, typename dist_t>
    inline vertex_t VPTree<T, dist_t>::get_random_unvisited_vertex() const
    {
        if(empty())
            throw VPTreeError("VPTree", "get_random_unvisited_vertex", "No remaining vertices");

        std::size_t idx = static_cast<std::size_t>(RNG::rand_u64(0, remaining_vertices.size()));
        return remaining_vertices[idx];
    }

    template <typename T, typename dist_t>
    inline std::vector<const T*> VPTree<T, dist_t>::get_remaining_elements() const
    {
        std::vector<const T*> remaining_elements_ptr;
        remaining_elements_ptr.resize(remaining_vertices.size());

        for(std::size_t i = 0; i < remaining_vertices.size(); ++i)
            remaining_elements_ptr[i] = elements[remaining_vertices[i]];

        return remaining_elements_ptr;
    }

    template <typename T, typename dist_t>
    inline typename VPTree<T, dist_t>::VPTreeNode* VPTree<T, dist_t>::init_node(std::vector<vertex_t>::iterator begin, std::vector<vertex_t>::iterator end, VPTreeNode* parent)
    {
        //leaf
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
            std::vector<dist_t> distances;
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

    template <typename T, typename dist_t>
    inline typename VPTree<T, dist_t>::partition_result_t VPTree<T, dist_t>::partition_vertices_by_median_distance(std::vector<vertex_t>::iterator vertices_begin, std::vector<vertex_t>::iterator vertices_end, const std::vector<dist_t>& distances)
    {
        const std::size_t size = static_cast<std::size_t>(std::distance(vertices_begin, vertices_end));

        if (size == 0)
            throw VPTreeError("VPTree", "partition_vertices_by_median_distance", "got empty range");

        //Zip into pairs so both vectors move together.
        std::vector<std::pair<dist_t, vertex_t>> pairs;
        pairs.reserve(size);
        for (std::size_t i = 0; i < size; ++i)
            pairs.emplace_back(distances[i], std::move(vertices_begin[i]));

        auto byFirst = [](const auto& a, const auto& b) { return a.first < b.first; };

        //Introselect
        auto mid = pairs.begin() + size / 2;
        std::nth_element(pairs.begin(), mid, pairs.end(), byFirst);

        dist_t median;
        if (size % 2 == 1) {
            median = mid->first; // odd size: exact middle element
        } else {
            // even size: average of the two middle elements.
            // mid->first is the upper median
            // The lower median is the max of everything before mid.
            dist_t upper = mid->first;
            dist_t lower = std::max_element(pairs.begin(), mid, byFirst)->first;
            median = (lower + upper) / dist_t{2};
        }

        auto splitPoint = std::partition(
            pairs.begin(), pairs.end(),
            [median](const auto& p) { return p.first < median; }
        );
        std::size_t split_index = static_cast<std::size_t>(std::distance(pairs.begin(), splitPoint));

        //Write back vertices
        for (std::size_t i = 0; i < size; ++i)
            vertices_begin[i] = std::move(pairs[i].second);

        return { split_index, median };
    }


    template <typename T, typename dist_t>
    inline std::size_t VPTree<T, dist_t>::remaining_size() const
    {
        return remaining_vertices.size();
    }

    template <typename T, typename dist_t>
    inline void VPTree<T, dist_t>::set_all_vertices_as_unvisited()
    {
        const std::size_t n = size();

        for(std::size_t i = 0; i < n; i++)
        {
            nodes[i].skip = false;
            already_added_vertices[i] = false;
            remaining_vertices[i] = vertex_t{i};
        }

        //Remaining vertices need to be randomly permuted to ensure proper random initial solutions
        std::shuffle(remaining_vertices.begin(), remaining_vertices.end(), RNG::gen);

        //Track new position of elements
        for(std::size_t i = 0; i < n; i++)
            remaining_vertices_position[remaining_vertices[i]] = i;
    }

    template <typename T, typename dist_t>
    inline void VPTree<T, dist_t>::set_vertex_as_unvisited(vertex_t vertex)
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

    template <typename T, typename dist_t>
    inline void VPTree<T, dist_t>::set_vertex_as_visited(vertex_t vertex)
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

    template <typename T, typename dist_t>
    inline std::size_t VPTree<T, dist_t>::size() const
    {
        return nodes.size();
    }

    // RNG implementation
    inline std::uint64_t RNG::rand_u64()
    {
        return gen();
    }

    inline std::uint64_t RNG::rand_u64(std::uint64_t a, std::uint64_t b) //Generate an unsigned integer in [a ; b[
    {
        if(a >= b)
            throw VPTreeError("RNG", "rand_u64", "got unexpected interval");

        return (rand_u64() % (b-a)) + a;
    }

    inline std::uint64_t RNG::get_seed()
    {
        return seed;
    }

    inline void RNG::set_seed(std::uint64_t new_seed)
    {
        seed = new_seed;
        gen.seed(seed);
    }

    inline std::uint64_t RNG::get_random_seed()
    {
        return std::random_device()();
    }

    constexpr inline std::uint64_t RNG::seed = std::uint64_t{42};
    inline std::mt19937 RNG::gen = std::mt19937(RNG::seed); // Standard mersenne_twister_engine seeded with default random_device
};

#endif