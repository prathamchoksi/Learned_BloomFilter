#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "bloom_filter.hpp"
#include "fused_trigram_scorer.hpp"

namespace py = pybind11;

PYBIND11_MODULE(learned_bloom_cpp, m)
{
    m.doc() = "C++ Bloom filter bindings";

    py::class_<BloomFilter>(m, "BloomFilter")
        .def(py::init<std::uint64_t, std::uint32_t>())
        .def("add", &BloomFilter::add)
        .def("contains", &BloomFilter::contains);

    py::class_<FusedTrigramScorer>(m, "FusedTrigramScorer")
        .def(py::init<const std::string &>())
        .def("score", &FusedTrigramScorer::score)
        .def("predict", &FusedTrigramScorer::predict, py::arg("url"), py::arg("threshold") = 0.5);
}
