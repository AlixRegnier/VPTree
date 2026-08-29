# Fast nearest-neighbor

## When using this library

Your distance function **must** satisfies mathematical distance properties:
|<!-- -->|<!-- -->|
:--|:--
$d(x, y) \ge 0$ | non-negativity
$d(x, y) = 0 \Leftrightarrow x = y$| identity of indiscernibles
$d(x, y) = d(y, x)$ | symmetric
$d(x, z) \le d(x, y) + d(y, z)$ | triangle inequality
    
Example of valid distances: Euclidean (L2), Manhattan (L1), Jaccard, Hamming, ...  
*Note: Cosine do not satisfies triangle inequality*

**Whenever you need:**
* Nearest-Neighbor Search (NNS) with ``epsilon = 0.0``
* Approximate Nearest-Neighbor Search (ANNS) with ``epsilon > 0.0``
* 1-NN queries
* $k$-NN queries
* Nearest-neighbor **without replacement**  
(perfect for  solving nearest-neighbor heuristic)

## Requirements

This library is a standalone header-only that requires at least ``C++17``.

It also has a Python interface, wrapped with [nanobind](https://github.com/wjakob/nanobind).
Python package be installed using:
```bash
python3 -m pip install git+https://github.com/AlixRegnier/VPTree.git
```
## Usage

We give examples in Python and in C++ in ``./examples``.

### Initialization

```c++
#include <vptree.hpp>

int main()
{
    std::vector<YourObject> elements = { ... };

    //Distance function must satisfies mathematical 
    //properties given in first section
    auto dist_func = [](const YourObject& a, const YourObject& b) -> double {
        //return distance between a and b
    };

    //Default type for distance is double
    //Only one constructor that takes two iterators and a distance function
    vptree::VPTree<YourObject, double> metric_tree(elements.begin(), elements.end(), dist_func);
}
```

### Communicating with library

VPTree internal operations relies on a type called ``vertex_t`` (alias ``std::uint64_t``).  

The $N$ elements you give when constructing a VPTree are mapped using an incremental ID:  
First element ID is $0$, second element ID is $1$, ..., last element ID is $N-1$.


Following methods take a ``vertex_t`` as input.
```c++
//Tells to VPTree to remove given vertex from search space
void set_vertex_as_visited(vertex_t vertex);

//Tells to VPTree to put back given vertex in search space
void set_vertex_as_unvisited(vertex_t vertex);

//Tells to VPTree to put back all vertices in search space
void set_all_vertices_as_unvisited();

//Retrieve element pointer associated with given vertex
const T* get_element_from_vertex(vertex_t vertex) const;
```

*Note: their is no mapping [T] --> [vertex], which avoid using a heavy map*.

### Queries

Queries are solved using a modified VPTree branch-and-bound algorithm. It introduces dynamic tree masking and approximated results.

``epsilon`` ($\epsilon$) can take any value in interval [0.0; +$\infty$]. By setting $\epsilon=0.0$, you will get exact nearest neighbor. Otherwise, you will get an approximate nearest neighbor. With $\tau$ being the true best distance, bound relaxation formula is:
$$\tilde\tau = \tau\cdot(1+\epsilon)$$

A result is represented by a template struct ``nn_t<T, dist_t>`` which contains following elements:
* ``const T* element_ptr`` : pointer to found element
* ``vertex_t vertex`` : corresponding vertex ID
* ``dist_t distance`` : distance from found element to query

```c++
//Returns the nearest unvisited neighbor to 'query' using epsilon as error factor
//Time complexity: O(log(n)), worst-case: O(n)
nn_t<T, dist_t> get_nearest_unvisited_neighbor(const T& query, double epsilon = 0.0) const;

//Returns a vector containing the k nearest unvisited neighbors to 'query' using epsilon as error factor
//Time complexity: O(k.(log(n)+log(k)), worst-case: O((k^2)*n)
std::vector<nn_t<T, dist_t>> get_k_nearest_unvisited_neighbors(const T& query, std::size_t k, double epsilon = 0.0) const;
```
### Misc

Some other functions that can be useful.
```c++
//Returns a random unvisited element in O(1) time.
const T* get_random_unvisited_element() const;

//Returns a vector containing k random unvisited elements in O(k) time.
std::vector<const T*> get_k_random_unvisited_elements() const;

//Returns a random unvisited vertex ID in O(1) time.
vertex_t get_random_unvisited_vertex() const;

//Returns a vector containing k random unvisited vertices in O(k) time.
std::vector<vertex_t> get_k_random_unvisited_vertices() const;

//Get remaining elements ptr (copy, may not be efficient)
std::vector<const T*> get_remaining_elements() const;
```
*Note: RNG seed can be modified using:* ``vptree::RNG::set_seed(std::uint64_t)``

## Implementation

* Uses STL vectors as underlying structure to contain the binary metric tree.
* $k$ methods are implemented using datastructures adapted to $k\le 15$.
* Memory leak free: all dynamic allocations rely on ``std::vector``.
* Distance function is stored once in a ``std::function``.
* Uses no maps and no sets.
* Indexed input elements are stored as pointers.
* Elements can be excluded from search space in $\mathcal{O}(\log n)$.
* Any exclusion can be reverted in $\mathcal{O}(\log n)$.
* Remaining elements are maintened in $\mathcal{O}(1)$ time complexity with a swap-and-pop algorithm.
* During construction, median is found in $\mathcal{O}(n)$ on average by C++ STL introselect (``nth_element``).
* Uses the MT19937 engine from C++ STL implementation with a default set to seed 42.
* All VPTree methods are virtual if you need to extense their behavior.

## References

[1] Yianilos, Peter N. "Data structures and algorithms for nearest neighbor search in general metric spaces." Soda. Vol. 93. No. 194. 1993.
[2] preprint soon
