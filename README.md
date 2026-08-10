# Fast nearest-neighbor

## When using this library

* 1NN query (only one candidate)
* Nearest-Neighbor (NN) with ``epsilon = 0.0``
* Approximate Nearest-Neighbor (ANN) with ``epsilon > 0.0``
* Nearest-neighbor peeling **without replacement**  
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

### Communicating with library

VPTree internal operations relies on a type call ``vertex_t`` (alias ``std::uint64_t``).  

The $N$ elements you give when constructing a VPTree are mapped using an incremental ID:  
First element ID is $0$, second element ID is $1$, ..., last element ID is $N-1$.


Following methods take a ``vertex_t`` as input.
```c++
//Tells to VPTree to remove given vertex from search space
void set_vertex_as_visited(vertex_t vertex);

//Tells to VPTree to put back given vertex in search space
void set_vertex_as_unvisited(vertex_t vertex);

//Retrieve element pointer associated with given vertex
const T* get_element_from_vertex(vertex_t vertex) const;
```

*Note: their is no mapping [T] --> [vertex], which avoid using a heavy map*.

### Queries

Queries are solved using a modified VPTree branch-and-bound algorithm. It introduces dynamic tree masking and approximated results.

``epsilon`` ($\epsilon$) can take any value in interval [0.0; +$\infty$]. By setting $\epsilon=0.0$, you will get exact nearest neighbor. Otherwise, you will get an approximate nearest neighbor. With $\tau$ being the true best distance, bound relaxation formula is:
$$\tilde\tau = \tau\cdot(1+\epsilon)^{-1}$$

A result is represented by a template struct ``nn_t<T>`` which contains following elements:
* ``const T* element_ptr`` : pointer to found element
* ``vertex_t vertex`` : corresponding vertex ID
* ``double distance`` : distance from found element to query

```c++
//Returns the nearest unvisited neighbor to 'query' using epsilon as error factor
nn_t<T> get_nearest_unvisited_neighbor(const T& query, double epsilon = 0.0) const;
```
### Misc

Some other functions that can be useful.
```c++
//Returns a random unvisited element in O(1) time.
const T* get_random_unvisited_element() const;

//Returns a random unvisited vertex ID in O(1) time.
vertex_t get_random_unvisited_vertex() const;

//Get remaining elements ptr (copy, may not be efficient)
std::vector<const T*> get_remaining_elements_ptr() const;
```
*Note: RNG seed can be modified using:* ``vptree::RNG::set_seed(std::uint64_t)``

## Implementation

* Uses a VPTree [1] as underlying structure (binary metric tree).
* Memory leak free: all dynamic allocations rely on ``std::vector``.
* Uses no maps and no sets.
* Indexed input elements are stored as pointers.
* Remaining elements are maintened in $\mathcal{O}(1)$ time complexity with a swap-and-pop algorithm.
* Median is found in $\mathcal{O}(n)$ on average by C++ STL introselect (``nth_element``).
* Uses the MT19937 from C++ STL implementation with a default set to seed 42.
* All VPTree methods are virtual if you need to extense their behavior.


## Not implemented yet 

**KNN**.

## References

[1] ref vptree  
[2] ref relaxed vptree
