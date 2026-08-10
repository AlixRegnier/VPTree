// binding.cpp
//
// nanobind bindings for the vptree::VPTree<T> template class.
//
// Since VPTree, nn_t, DistanceFunction, etc. are templates, we instantiate
// and expose a single concrete specialization for T = nb::object. This lets
// Python code build a VPTree out of arbitrary Python objects, supplying an
// arbitrary Python callable as the distance function.
//
// Notes on the wrapping strategy:
//   - VPTree<T>::elements is a vector of raw pointers (const T*) into
//     externally-owned storage. To keep that storage alive for the whole
//     lifetime of the tree, PyVPTree owns a std::vector<nb::object> and
//     builds the VPTree from iterators into that vector. The vector is
//     never resized after construction, so the pointers VPTree stores
//     remain valid.
//   - VPTree's real constructor is a template on the iterator type, so it
//     cannot be bound directly; PyVPTree's constructor is the concrete,
//     bindable entry point that performs the templated call internally.
//   - vptree::VPTreeError is translated to a Python exception derived from
//     RuntimeError.

#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "../include/vptree.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace
{
    // Plain-data mirror of vptree::nn_t<nb::object> for the Python side:
    // nn_t<T> stores a raw pointer to the element, which we dereference
    // into an owned nb::object before crossing back into Python.
    struct PyNN
    {
        nb::object element;
        vptree::vertex_t vertex;
        double distance;
    };

    // Concrete wrapper around VPTree<nb::object>.
    //
    // Owns the backing storage for the elements so that the pointers held
    // internally by VPTree stay valid for as long as the Python object
    // wrapping this class is alive.
    class PyVPTree
    {
    public:
        PyVPTree(std::vector<nb::object> elements,
                 vptree::DistanceFunction<nb::object> dist_func)
            : storage(std::move(elements)),
              tree(storage.begin(), storage.end(), dist_func)
        {
        }

        void set_vertex_as_visited(vptree::vertex_t vertex)
        {
            tree.set_vertex_as_visited(vertex);
        }

        void set_vertex_as_unvisited(vptree::vertex_t vertex)
        {
            tree.set_vertex_as_unvisited(vertex);
        }

        nb::object get_element_from_vertex(vptree::vertex_t vertex) const
        {
            const nb::object* ptr = tree.get_element_from_vertex(vertex);
            return ptr ? *ptr : nb::object(nb::none());
        }

        PyNN get_nearest_unvisited_neighbor(const nb::object& query,
                                             double epsilon = 0.0) const
        {
            vptree::nn_t<nb::object> result =
                tree.get_nearest_unvisited_neighbor(query, epsilon);

            return PyNN{
                result.element_ptr ? *result.element_ptr : nb::object(nb::none()),
                result.vertex,
                result.distance
            };
        }

        nb::object get_random_unvisited_element() const
        {
            const nb::object* ptr = tree.get_random_unvisited_element();
            return ptr ? *ptr : nb::object(nb::none());
        }

        vptree::vertex_t get_random_unvisited_vertex() const
        {
            return tree.get_random_unvisited_vertex();
        }

        bool empty() const
        {
            return tree.empty();
        }

        std::size_t size() const
        {
            return tree.size();
        }

        std::size_t remaining_size() const
        {
            return tree.remaining_size();
        }

        std::vector<nb::object> get_remaining_elements_ptr() const
        {
            std::vector<const nb::object*> ptrs = tree.get_remaining_elements_ptr();

            std::vector<nb::object> result;
            result.reserve(ptrs.size());
            for (const nb::object* p : ptrs)
                result.push_back(p ? *p : nb::object(nb::none()));

            return result;
        }

    private:
        // Declaration order matters: storage must outlive / be constructed
        // before tree, since tree's iterators point into storage.
        std::vector<nb::object> storage;
        vptree::VPTree<nb::object> tree;
    };
}

NB_MODULE(vptree, m)
{
/*
    Currently no memory leaks have been found in library, nothing is instancied on the heap manually.
    nanobind's hook "atexit" warns about possible memory just before GC free memory.

    To reproduce, pass "true" to "nb::set_leak_warnings()" and in a Python script:
    >>>
    from vptree import VPTree
    tree = VPTree(elements, lambda a,b: 0.0)
    <<<

    If VPTree is instancied in another scope than the global scope, it do not produce any warning from nanobind
*/
    nb::set_leak_warnings(false);

    m.doc() = "Python bindings for the vptree::VPTree<T> vantage-point tree "
               "(instantiated for arbitrary Python objects)";

    // --- Exception translation -------------------------------------------
    nb::exception<vptree::VPTreeError>(m, "VPTreeError", PyExc_RuntimeError);

    // --- nn_t<nb::object> mirror ------------------------------------------
    nb::class_<PyNN>(m, "NearestNeighbor")
        .def_ro("element", &PyNN::element, "The nearest neighbor's element")
        .def_ro("vertex", &PyNN::vertex, "The nearest neighbor's vertex id")
        .def_ro("distance", &PyNN::distance,
                "Distance between the query and the nearest neighbor")
        .def("__repr__", [](const PyNN& self) {
            return "<NearestNeighbor vertex=" + std::to_string(self.vertex) +
                   " distance=" + std::to_string(self.distance) + ">";
        });

    // --- VPTree<nb::object> ------------------------------------------------
    nb::class_<PyVPTree>(m, "VPTree")
        .def(nb::init<std::vector<nb::object>, vptree::DistanceFunction<nb::object>>(),
             "elements"_a, "dist_func"_a,
             "Build a vantage-point tree from a list of Python objects and a "
             "distance function dist_func(a, b) -> float")

        .def("set_vertex_as_visited", &PyVPTree::set_vertex_as_visited,
             "vertex"_a,
             "Mark the given vertex as visited (excluded from future "
             "nearest-neighbor queries)")

        .def("set_vertex_as_unvisited", &PyVPTree::set_vertex_as_unvisited,
             "vertex"_a,
             "Mark the given vertex as unvisited (eligible again for "
             "nearest-neighbor queries)")

        .def("get_element_from_vertex", &PyVPTree::get_element_from_vertex,
             "vertex"_a, "Return the element associated with a vertex id")

        .def("get_nearest_unvisited_neighbor",
             &PyVPTree::get_nearest_unvisited_neighbor,
             "query"_a, "epsilon"_a = 0.0,
             "Return the nearest unvisited neighbor to query")

        .def("get_random_unvisited_element",
             &PyVPTree::get_random_unvisited_element,
             "Return a random element among the unvisited ones")

        .def("get_random_unvisited_vertex",
             &PyVPTree::get_random_unvisited_vertex,
             "Return a random vertex id among the unvisited ones")

        .def("get_remaining_elements_ptr",
             &PyVPTree::get_remaining_elements_ptr,
             "Return the list of all currently unvisited elements")

        .def("empty",
             &PyVPTree::empty,
             "Tell whether there are remaining elements or not")

        .def("size",
             &PyVPTree::size,
             "Return the number of indexed elements")

        .def("remaining_size",
             &PyVPTree::remaining_size,
             "Return the number of remaining elements");

    // --- RNG -----------------------------------------------------------
    // RNG's constructor is deleted and every member is static, so we bind
    // it purely as a namespace of static functions (no __init__ exposed).
    nb::class_<vptree::RNG>(m, "RNG")
        .def_static("rand_u64",
                    nb::overload_cast<>(&vptree::RNG::rand_u64),
                    "Return a random uint64")
        .def_static("rand_u64_range",
                    nb::overload_cast<std::uint64_t, std::uint64_t>(
                        &vptree::RNG::rand_u64),
                    "a"_a, "b"_a,
                    "Return a random uint64 in the inclusive range [a, b]")
        .def_static("set_seed", &vptree::RNG::set_seed, "seed"_a,
                    "Set the RNG seed")
        .def_static("get_seed", &vptree::RNG::get_seed,
                    "Return the current RNG seed")
        .def_static("get_random_seed", &vptree::RNG::get_random_seed,
                    "Return a new random number that can be used as a seed");
}