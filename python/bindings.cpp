#include "../include/vptree.h"  // adjust to your actual header name

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <memory>

namespace nb = nanobind;
using namespace nb::literals;

using VPTreeObj = VPTree<nb::object>;
using NN = nn_t<nb::object>;

class PyVPTree {
public:
    PyVPTree(nb::sequence elements, nb::callable dist_func) {
        storage_.reserve(nb::len(elements));
        for (nb::handle h : elements) {
            storage_.push_back(nb::borrow<nb::object>(h));
        }

        DistanceFunction<nb::object> df =
            [dist_func](const nb::object& a, const nb::object& b) -> double {
                return nb::cast<double>(dist_func(a, b));
            };

        // storage_ is fixed-size from here on, so T* pointers into it stay valid
        tree_ = std::make_unique<VPTreeObj>(storage_.begin(), storage_.end(), df);
    }

    void set_vertex_as_visited(vertex_t v) { tree_->set_vertex_as_visited(v); }
    void set_vertex_as_unvisited(vertex_t v) { tree_->set_vertex_as_unvisited(v); }

    nb::tuple get_nearest_unvisited_neighbor(const nb::object& query, double epsilon = 0.0) const {
        NN result = tree_->get_nearest_unvisited_neighbor(query, epsilon);
        nb::object elem = result.element ? *result.element : nb::none();
        return nb::make_tuple(elem, result.vertex, result.distance);
    }

    vertex_t get_random_unvisited_vertex() const {
        return tree_->get_random_unvisited_vertex();
    }

private:
    std::vector<nb::object> storage_;
    std::unique_ptr<VPTreeObj> tree_;
};

NB_MODULE(vptree_ext, m) {
    nb::class_<PyVPTree>(m, "VPTree")
        .def(nb::init<nb::sequence, nb::callable>(), "elements"_a, "dist_func"_a)
        .def("set_vertex_as_visited", &PyVPTree::set_vertex_as_visited, "vertex"_a)
        .def("set_vertex_as_unvisited", &PyVPTree::set_vertex_as_unvisited, "vertex"_a)
        .def("get_nearest_unvisited_neighbor",
             &PyVPTree::get_nearest_unvisited_neighbor,
             "query"_a, "epsilon"_a = 0.0)
        .def("get_random_unvisited_vertex", &PyVPTree::get_random_unvisited_vertex);
}
/*
NB_MODULE(vptree, m)
{
    /*
        Currently no memory leaks have been found in library, nothing is instancied on the heap manually.
        nanobind's hook "atexit" warns about possible memory just before GC free memory.

        To reproduce, pass "true" to "nb::set_leak_warnings()" and in a Python script:
>>>
from vptree import VPTree
tree = VPTree(1, lambda a,b: 0.0)
<<<
    
    nb::set_leak_warnings(false);

    // ---- nn_t struct ----
    nb::class_<vptree::nn_t<nb::object>>(m, "NearestNeighbor")
        .def_ro("vertex", &vptree::nn_t::vertex)
        .def_ro("distance", &vptree::nn_t::distance)
        .def("__repr__", [](const vptree::nn_t& self) {
            return "NearestNeighbor(vertex=" + std::to_string(self.vertex) +
                   ", distance=" + std::to_string(self.distance) + ")";
        });

    // ---- VPTreeError ----
    nb::exception<vptree::VPTreeError>(m, "VPTreeError", PyExc_RuntimeError);

    // ---- VPTree ----
    nb::class_<vptree::VPTree>(m, "VPTree", nb::is_final())
        .def(
            nb::init<std::size_t, const vptree::DistanceFunction&>(),
            "n"_a, "dist_func"_a,
            "Construct a VPTree over n vertices [0, n) using the given distance function."
        )

        .def(
            "set_vertex_as_visited",
            &vptree::VPTree::set_vertex_as_visited,
            "vertex"_a
        )

        .def(
            "set_vertex_as_unvisited",
            &vptree::VPTree::set_vertex_as_unvisited,
            "vertex"_a
        )

        .def(
            "get_nearest_unvisited_neighbor",
            static_cast<vptree::nn_t (vptree::VPTree::*)(vptree::vertex_t, double) const>(
                &vptree::VPTree::get_nearest_unvisited_neighbor
            ),
            "query"_a, "epsilon"_a = 0.0
        )

        .def(
            "get_random_unvisited_vertex",
            &vptree::VPTree::get_random_unvisited_vertex
        );
}*/