// HPXPy - Algorithm bindings
//
// SPDX-License-Identifier: BSL-1.0

#include "ndarray.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <hpx/algorithm.hpp>
#include <hpx/execution.hpp>
#include <hpx/numeric.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>

namespace py = pybind11;

namespace hpxpy {

// Dispatch helper for typed operations
template<typename Func>
auto dispatch_dtype(ndarray const& arr, Func&& func) {
    char kind = arr.dtype().kind();
    auto itemsize = arr.dtype().itemsize();

    if (kind == 'f') {  // floating point
        if (itemsize == 8) {
            return func(static_cast<double const*>(arr.data()), arr.size());
        } else if (itemsize == 4) {
            return func(static_cast<float const*>(arr.data()), arr.size());
        }
    } else if (kind == 'i') {  // signed integer
        if (itemsize == 8) {
            return func(static_cast<int64_t const*>(arr.data()), arr.size());
        } else if (itemsize == 4) {
            return func(static_cast<int32_t const*>(arr.data()), arr.size());
        }
    } else if (kind == 'u') {  // unsigned integer
        if (itemsize == 8) {
            return func(static_cast<uint64_t const*>(arr.data()), arr.size());
        } else if (itemsize == 4) {
            return func(static_cast<uint32_t const*>(arr.data()), arr.size());
        }
    }

    throw std::runtime_error("Unsupported dtype: " +
        py::str(arr.dtype()).cast<std::string>());
}

// Sum reduction
py::object sum(std::shared_ptr<ndarray> arr) {
    if (arr->size() == 0) {
        return py::cast(0.0);
    }

    // Release GIL during computation
    py::gil_scoped_release release;

    return dispatch_dtype(*arr, [](auto const* data, py::ssize_t size) -> py::object {
        using T = std::remove_const_t<std::remove_pointer_t<decltype(data)>>;

        // Phase 1: Use sequential execution
        // Phase 2 will add parallel execution policies
        T result = hpx::reduce(
            hpx::execution::seq,
            data, data + size,
            T{0}
        );

        py::gil_scoped_acquire acquire;
        return py::cast(result);
    });
}

// Product reduction
py::object prod(std::shared_ptr<ndarray> arr) {
    if (arr->size() == 0) {
        return py::cast(1.0);
    }

    py::gil_scoped_release release;

    return dispatch_dtype(*arr, [](auto const* data, py::ssize_t size) -> py::object {
        using T = std::remove_const_t<std::remove_pointer_t<decltype(data)>>;

        T result = hpx::reduce(
            hpx::execution::seq,
            data, data + size,
            T{1},
            std::multiplies<T>{}
        );

        py::gil_scoped_acquire acquire;
        return py::cast(result);
    });
}

// Minimum reduction
py::object min(std::shared_ptr<ndarray> arr) {
    if (arr->size() == 0) {
        throw std::runtime_error("min() arg is an empty sequence");
    }

    py::gil_scoped_release release;

    return dispatch_dtype(*arr, [](auto const* data, py::ssize_t size) -> py::object {
        using T = std::remove_const_t<std::remove_pointer_t<decltype(data)>>;

        auto result = hpx::min_element(
            hpx::execution::seq,
            data, data + size
        );

        py::gil_scoped_acquire acquire;
        return py::cast(*result);
    });
}

// Maximum reduction
py::object max(std::shared_ptr<ndarray> arr) {
    if (arr->size() == 0) {
        throw std::runtime_error("max() arg is an empty sequence");
    }

    py::gil_scoped_release release;

    return dispatch_dtype(*arr, [](auto const* data, py::ssize_t size) -> py::object {
        using T = std::remove_const_t<std::remove_pointer_t<decltype(data)>>;

        auto result = hpx::max_element(
            hpx::execution::seq,
            data, data + size
        );

        py::gil_scoped_acquire acquire;
        return py::cast(*result);
    });
}

// Sort (returns new sorted array)
std::shared_ptr<ndarray> sort(std::shared_ptr<ndarray> arr) {
    if (arr->ndim() != 1) {
        throw std::runtime_error("sort only supports 1D arrays in Phase 1");
    }

    // Create a copy for sorting
    auto result = std::make_shared<ndarray>(arr->shape(), arr->dtype());
    std::memcpy(result->data(), arr->data(), arr->nbytes());

    // Release GIL during sort
    py::gil_scoped_release release;

    char kind = arr->dtype().kind();
    auto itemsize = arr->dtype().itemsize();

    if (kind == 'f') {
        if (itemsize == 8) {
            auto* ptr = static_cast<double*>(result->data());
            hpx::sort(hpx::execution::seq, ptr, ptr + arr->size());
        } else if (itemsize == 4) {
            auto* ptr = static_cast<float*>(result->data());
            hpx::sort(hpx::execution::seq, ptr, ptr + arr->size());
        }
    } else if (kind == 'i') {
        if (itemsize == 8) {
            auto* ptr = static_cast<int64_t*>(result->data());
            hpx::sort(hpx::execution::seq, ptr, ptr + arr->size());
        } else if (itemsize == 4) {
            auto* ptr = static_cast<int32_t*>(result->data());
            hpx::sort(hpx::execution::seq, ptr, ptr + arr->size());
        }
    } else if (kind == 'u') {
        if (itemsize == 8) {
            auto* ptr = static_cast<uint64_t*>(result->data());
            hpx::sort(hpx::execution::seq, ptr, ptr + arr->size());
        } else if (itemsize == 4) {
            auto* ptr = static_cast<uint32_t*>(result->data());
            hpx::sort(hpx::execution::seq, ptr, ptr + arr->size());
        }
    } else {
        py::gil_scoped_acquire acquire;
        throw std::runtime_error("Unsupported dtype for sort");
    }

    return result;
}

// Count occurrences of a value
py::ssize_t count(std::shared_ptr<ndarray> arr, py::object value) {
    char kind = arr->dtype().kind();
    auto itemsize = arr->dtype().itemsize();

    if (kind == 'f') {
        if (itemsize == 8) {
            double val = value.cast<double>();
            py::gil_scoped_release release;
            auto* ptr = static_cast<double const*>(arr->data());
            return hpx::count(hpx::execution::seq, ptr, ptr + arr->size(), val);
        } else if (itemsize == 4) {
            float val = value.cast<float>();
            py::gil_scoped_release release;
            auto* ptr = static_cast<float const*>(arr->data());
            return hpx::count(hpx::execution::seq, ptr, ptr + arr->size(), val);
        }
    } else if (kind == 'i') {
        if (itemsize == 8) {
            int64_t val = value.cast<int64_t>();
            py::gil_scoped_release release;
            auto* ptr = static_cast<int64_t const*>(arr->data());
            return hpx::count(hpx::execution::seq, ptr, ptr + arr->size(), val);
        } else if (itemsize == 4) {
            int32_t val = value.cast<int32_t>();
            py::gil_scoped_release release;
            auto* ptr = static_cast<int32_t const*>(arr->data());
            return hpx::count(hpx::execution::seq, ptr, ptr + arr->size(), val);
        }
    }

    throw std::runtime_error("Unsupported dtype for count");
}

}  // namespace hpxpy

void bind_algorithms(py::module_& m) {
    m.def("_sum", &hpxpy::sum,
        py::arg("arr"),
        R"pbdoc(
            Sum of array elements.

            Parameters
            ----------
            arr : ndarray
                Input array.

            Returns
            -------
            scalar
                Sum of all elements.
        )pbdoc");

    m.def("_prod", &hpxpy::prod,
        py::arg("arr"),
        "Product of array elements.");

    m.def("_min", &hpxpy::min,
        py::arg("arr"),
        "Minimum of array elements.");

    m.def("_max", &hpxpy::max,
        py::arg("arr"),
        "Maximum of array elements.");

    m.def("_sort", &hpxpy::sort,
        py::arg("arr"),
        "Return a sorted copy of the array.");

    m.def("_count", &hpxpy::count,
        py::arg("arr"), py::arg("value"),
        "Count occurrences of a value.");
}

void bind_execution(py::module_& m) {
    // Create execution submodule
    auto exec = m.def_submodule("execution",
        "Execution policies for controlling parallel execution.");

    // For Phase 1, we just define placeholders for execution policies
    // Phase 2 will implement full policy support

    py::class_<hpx::execution::sequenced_policy>(exec, "sequenced_policy",
        "Sequential execution policy.")
        .def(py::init<>());

    exec.attr("seq") = hpx::execution::seq;

    // Placeholder for parallel policies (Phase 2)
    exec.attr("par") = hpx::execution::seq;  // Fallback to seq for now
    exec.attr("par_unseq") = hpx::execution::seq;  // Fallback to seq for now
}
