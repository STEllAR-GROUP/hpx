// HPXPy - Distributed Array Bindings
//
// SPDX-License-Identifier: BSL-1.0
//
// Python bindings for distributed arrays.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "../types/distributed_array.hpp"

namespace py = pybind11;

namespace hpxpy {

// Helper to create distributed array from numpy
template<typename T>
distributed_array<T> from_numpy_distributed(
    py::array_t<T> arr,
    py::object distribution,
    std::size_t /*num_partitions*/)
{
    auto policy = parse_distribution(distribution);

    // Get shape
    std::vector<py::ssize_t> shape(arr.ndim());
    for (py::ssize_t i = 0; i < arr.ndim(); ++i) {
        shape[i] = arr.shape(i);
    }

    // Copy data to vector
    std::vector<T> data(arr.data(), arr.data() + arr.size());

    // Create distributed array
    return distributed_array<T>(data, shape, policy);
}

// Helper to create zeros
template<typename T>
distributed_array<T> zeros_distributed(
    std::vector<py::ssize_t> shape,
    py::object distribution,
    std::size_t num_partitions)
{
    auto policy = parse_distribution(distribution);
    distributed_array<T> arr(shape, policy, num_partitions);
    arr.fill(T{0});
    return arr;
}

// Helper to create ones
template<typename T>
distributed_array<T> ones_distributed(
    std::vector<py::ssize_t> shape,
    py::object distribution,
    std::size_t num_partitions)
{
    auto policy = parse_distribution(distribution);
    distributed_array<T> arr(shape, policy, num_partitions);
    arr.fill(T{1});
    return arr;
}

// Helper to create full
template<typename T>
distributed_array<T> full_distributed(
    std::vector<py::ssize_t> shape,
    T value,
    py::object distribution,
    std::size_t num_partitions)
{
    auto policy = parse_distribution(distribution);
    distributed_array<T> arr(shape, policy, num_partitions);
    arr.fill(value);
    return arr;
}

// Bind a specific distributed array type
template<typename T>
void bind_distributed_array_type(py::module_& m, const char* name) {
    using DA = distributed_array<T>;

    py::class_<DA>(m, name)
        .def(py::init<std::vector<py::ssize_t>, DistributionPolicy, std::size_t>(),
             py::arg("shape"),
             py::arg("policy") = DistributionPolicy::None,
             py::arg("num_partitions") = 0,
             "Create a distributed array with given shape and distribution")

        // Properties
        .def_property_readonly("shape", &DA::shape,
            "Shape of the array")
        .def_property_readonly("size", &DA::size,
            "Total number of elements")
        .def_property_readonly("ndim", &DA::ndim,
            "Number of dimensions")
        .def_property_readonly("policy", &DA::policy,
            "Distribution policy")
        .def_property_readonly("num_partitions", &DA::num_partitions,
            "Number of partitions")
        .def_property_readonly("locality_id", &DA::locality_id,
            "ID of the locality that created this array")

        // Methods
        .def("is_distributed", &DA::is_distributed,
            "Check if array is actually distributed across localities")
        .def("to_numpy", &DA::to_numpy,
            "Convert to numpy array (gathers all data if distributed)")
        .def("fill", &DA::fill,
            py::arg("value"),
            "Fill array with a value")
        .def("get_distribution_info", &DA::get_distribution_info,
            "Get distribution information")

        // Repr
        .def("__repr__", [name](DA const& arr) {
            std::ostringstream oss;
            oss << name << "(shape=[";
            auto const& shape = arr.shape();
            for (size_t i = 0; i < shape.size(); ++i) {
                if (i > 0) oss << ", ";
                oss << shape[i];
            }
            oss << "], ";

            switch (arr.policy()) {
                case DistributionPolicy::None:
                    oss << "distribution='none'";
                    break;
                case DistributionPolicy::Block:
                    oss << "distribution='block'";
                    break;
                case DistributionPolicy::Cyclic:
                    oss << "distribution='cyclic'";
                    break;
            }
            oss << ", partitions=" << arr.num_partitions();
            if (arr.is_distributed()) {
                oss << ", distributed=True";
            }
            oss << ")";
            return oss.str();
        });
}

void register_distributed_array_bindings(py::module_& m) {
    // Note: DistributionPolicy enum is already registered in array_bindings.cpp
    // in the 'distribution' submodule. Don't re-register it here.

    // Distribution info struct
    py::class_<DistributionInfo>(m, "DistributionInfo",
        "Information about array distribution")
        .def_readonly("policy", &DistributionInfo::policy)
        .def_readonly("num_partitions", &DistributionInfo::num_partitions)
        .def_readonly("chunk_size", &DistributionInfo::chunk_size)
        .def_readonly("locality_id", &DistributionInfo::locality_id)
        .def("is_distributed", &DistributionInfo::is_distributed);

    // Bind distributed array types
    bind_distributed_array_type<double>(m, "DistributedArrayF64");
    bind_distributed_array_type<float>(m, "DistributedArrayF32");
    bind_distributed_array_type<std::int64_t>(m, "DistributedArrayI64");
    bind_distributed_array_type<std::int32_t>(m, "DistributedArrayI32");

    // Factory functions for distributed arrays
    m.def("distributed_zeros",
        [](std::vector<py::ssize_t> shape, py::object distribution, std::size_t num_partitions) {
            return zeros_distributed<double>(shape, distribution, num_partitions);
        },
        py::arg("shape"),
        py::arg("distribution") = py::none(),
        py::arg("num_partitions") = 0,
        "Create a distributed array filled with zeros");

    m.def("distributed_ones",
        [](std::vector<py::ssize_t> shape, py::object distribution, std::size_t num_partitions) {
            return ones_distributed<double>(shape, distribution, num_partitions);
        },
        py::arg("shape"),
        py::arg("distribution") = py::none(),
        py::arg("num_partitions") = 0,
        "Create a distributed array filled with ones");

    m.def("distributed_full",
        [](std::vector<py::ssize_t> shape, double value, py::object distribution, std::size_t num_partitions) {
            return full_distributed<double>(shape, value, distribution, num_partitions);
        },
        py::arg("shape"),
        py::arg("value"),
        py::arg("distribution") = py::none(),
        py::arg("num_partitions") = 0,
        "Create a distributed array filled with a value");

    m.def("distributed_from_numpy",
        [](py::array_t<double> arr, py::object distribution, std::size_t num_partitions) {
            return from_numpy_distributed<double>(arr, distribution, num_partitions);
        },
        py::arg("arr"),
        py::arg("distribution") = py::none(),
        py::arg("num_partitions") = 0,
        "Create a distributed array from a numpy array");
}

}  // namespace hpxpy
