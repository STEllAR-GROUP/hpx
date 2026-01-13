// HPXPy - Array bindings
//
// SPDX-License-Identifier: BSL-1.0

#include "ndarray.hpp"
#include "operators.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <hpx/hpx.hpp>

#include <cstring>
#include <memory>
#include <sstream>

namespace py = pybind11;

namespace hpxpy {

// Array creation functions
std::shared_ptr<ndarray> zeros(std::vector<py::ssize_t> shape, py::dtype dtype) {
    auto arr = std::make_shared<ndarray>(shape, dtype);
    std::memset(arr->data(), 0, arr->nbytes());
    return arr;
}

std::shared_ptr<ndarray> ones(std::vector<py::ssize_t> shape, py::dtype dtype) {
    auto arr = std::make_shared<ndarray>(shape, dtype);

    // Fill with ones based on dtype
    char kind = dtype.kind();
    if (kind == 'f') {  // floating point
        if (dtype.itemsize() == 4) {
            auto* ptr = arr->typed_data<float>();
            std::fill(ptr, ptr + arr->size(), 1.0f);
        } else if (dtype.itemsize() == 8) {
            auto* ptr = arr->typed_data<double>();
            std::fill(ptr, ptr + arr->size(), 1.0);
        }
    } else if (kind == 'i' || kind == 'u') {  // integer
        if (dtype.itemsize() == 4) {
            auto* ptr = arr->typed_data<int32_t>();
            std::fill(ptr, ptr + arr->size(), 1);
        } else if (dtype.itemsize() == 8) {
            auto* ptr = arr->typed_data<int64_t>();
            std::fill(ptr, ptr + arr->size(), 1);
        }
    }

    return arr;
}

std::shared_ptr<ndarray> empty(std::vector<py::ssize_t> shape, py::dtype dtype) {
    return std::make_shared<ndarray>(shape, dtype);
}

std::shared_ptr<ndarray> arange(double start, double stop, double step, py::object dtype_obj) {
    // Compute number of elements
    py::ssize_t n = static_cast<py::ssize_t>(std::ceil((stop - start) / step));
    if (n < 0) n = 0;

    // Determine dtype
    py::dtype dtype = dtype_obj.is_none()
        ? py::dtype::of<double>()
        : py::dtype::from_args(dtype_obj);

    auto arr = std::make_shared<ndarray>(std::vector<py::ssize_t>{n}, dtype);

    // Fill based on dtype
    char kind = dtype.kind();
    if (kind == 'f') {
        if (dtype.itemsize() == 8) {
            auto* ptr = arr->typed_data<double>();
            for (py::ssize_t i = 0; i < n; ++i) {
                ptr[i] = start + i * step;
            }
        } else if (dtype.itemsize() == 4) {
            auto* ptr = arr->typed_data<float>();
            for (py::ssize_t i = 0; i < n; ++i) {
                ptr[i] = static_cast<float>(start + i * step);
            }
        }
    } else if (kind == 'i') {
        if (dtype.itemsize() == 8) {
            auto* ptr = arr->typed_data<int64_t>();
            for (py::ssize_t i = 0; i < n; ++i) {
                ptr[i] = static_cast<int64_t>(start + i * step);
            }
        } else if (dtype.itemsize() == 4) {
            auto* ptr = arr->typed_data<int32_t>();
            for (py::ssize_t i = 0; i < n; ++i) {
                ptr[i] = static_cast<int32_t>(start + i * step);
            }
        }
    }

    return arr;
}

std::shared_ptr<ndarray> array_from_numpy(py::array np_array, bool copy) {
    return std::make_shared<ndarray>(np_array, copy);
}

}  // namespace hpxpy

void bind_array(py::module_& m) {
    // Bind the ndarray class
    py::class_<hpxpy::ndarray, std::shared_ptr<hpxpy::ndarray>>(m, "ndarray", py::buffer_protocol(),
        R"pbdoc(
            HPXPy N-dimensional array.

            This class represents a contiguous array of elements, similar to
            numpy.ndarray. It provides the foundation for parallel and
            distributed array operations.
        )pbdoc")
        .def_property_readonly("shape", [](hpxpy::ndarray const& arr) {
                return py::tuple(py::cast(arr.shape()));
            },
            "Shape of the array as a tuple.")
        .def_property_readonly("dtype", &hpxpy::ndarray::dtype,
            "Data type of array elements.")
        .def_property_readonly("size", &hpxpy::ndarray::size,
            "Total number of elements.")
        .def_property_readonly("ndim", &hpxpy::ndarray::ndim,
            "Number of dimensions.")
        .def_property_readonly("strides", [](hpxpy::ndarray const& arr) {
                return py::tuple(py::cast(arr.strides()));
            },
            "Strides of the array.")
        .def_property_readonly("nbytes", &hpxpy::ndarray::nbytes,
            "Total bytes consumed by array elements.")
        .def("to_numpy", &hpxpy::ndarray::to_numpy,
            "Convert to a NumPy array.")
        // Buffer protocol support
        .def_buffer([](hpxpy::ndarray& arr) -> py::buffer_info {
            return py::buffer_info(
                arr.data(),
                arr.dtype().itemsize(),
                std::string(1, arr.dtype().char_()),  // PEP 3118 format character
                arr.ndim(),
                arr.shape(),
                arr.strides()
            );
        })
        .def("__repr__", [](hpxpy::ndarray const& arr) {
            std::ostringstream ss;
            ss << "hpxpy.ndarray(shape=(";
            for (size_t i = 0; i < arr.shape().size(); ++i) {
                if (i > 0) ss << ", ";
                ss << arr.shape()[i];
            }
            ss << "), dtype=" << py::str(arr.dtype()).cast<std::string>() << ")";
            return ss.str();
        })
        // Arithmetic operators
        .def("__add__", [](hpxpy::ndarray const& a, py::object const& b) {
            return hpxpy::ops::dispatch_binary_py(a, b, hpxpy::ops::Add{});
        })
        .def("__radd__", [](hpxpy::ndarray const& a, py::object const& b) {
            return hpxpy::ops::dispatch_binary_py(a, b, hpxpy::ops::Add{});
        })
        .def("__sub__", [](hpxpy::ndarray const& a, py::object const& b) {
            return hpxpy::ops::dispatch_binary_py(a, b, hpxpy::ops::Sub{});
        })
        .def("__rsub__", [](hpxpy::ndarray const& a, py::object const& b) {
            return hpxpy::ops::dispatch_rbinary_py(a, b, hpxpy::ops::Sub{});
        })
        .def("__mul__", [](hpxpy::ndarray const& a, py::object const& b) {
            return hpxpy::ops::dispatch_binary_py(a, b, hpxpy::ops::Mul{});
        })
        .def("__rmul__", [](hpxpy::ndarray const& a, py::object const& b) {
            return hpxpy::ops::dispatch_binary_py(a, b, hpxpy::ops::Mul{});
        })
        .def("__truediv__", [](hpxpy::ndarray const& a, py::object const& b) {
            return hpxpy::ops::dispatch_binary_py(a, b, hpxpy::ops::Div{});
        })
        .def("__rtruediv__", [](hpxpy::ndarray const& a, py::object const& b) {
            return hpxpy::ops::dispatch_rbinary_py(a, b, hpxpy::ops::Div{});
        })
        .def("__floordiv__", [](hpxpy::ndarray const& a, py::object const& b) {
            return hpxpy::ops::dispatch_binary_py(a, b, hpxpy::ops::FloorDiv{});
        })
        .def("__rfloordiv__", [](hpxpy::ndarray const& a, py::object const& b) {
            return hpxpy::ops::dispatch_rbinary_py(a, b, hpxpy::ops::FloorDiv{});
        })
        .def("__mod__", [](hpxpy::ndarray const& a, py::object const& b) {
            return hpxpy::ops::dispatch_binary_py(a, b, hpxpy::ops::Mod{});
        })
        .def("__pow__", [](hpxpy::ndarray const& a, py::object const& b) {
            return hpxpy::ops::dispatch_binary_py(a, b, hpxpy::ops::Pow{});
        })
        // Unary operators
        .def("__neg__", [](hpxpy::ndarray const& a) {
            return hpxpy::ops::dispatch_unary(a, hpxpy::ops::Neg{});
        })
        .def("__pos__", [](hpxpy::ndarray const& a) {
            return hpxpy::ops::dispatch_unary(a, hpxpy::ops::Pos{});
        })
        .def("__abs__", [](hpxpy::ndarray const& a) {
            return hpxpy::ops::dispatch_unary(a, hpxpy::ops::Abs{});
        })
        // Comparison operators
        .def("__eq__", [](hpxpy::ndarray const& a, py::object const& b) {
            return hpxpy::ops::dispatch_compare(a, b, hpxpy::ops::Eq{});
        })
        .def("__ne__", [](hpxpy::ndarray const& a, py::object const& b) {
            return hpxpy::ops::dispatch_compare(a, b, hpxpy::ops::Ne{});
        })
        .def("__lt__", [](hpxpy::ndarray const& a, py::object const& b) {
            return hpxpy::ops::dispatch_compare(a, b, hpxpy::ops::Lt{});
        })
        .def("__le__", [](hpxpy::ndarray const& a, py::object const& b) {
            return hpxpy::ops::dispatch_compare(a, b, hpxpy::ops::Le{});
        })
        .def("__gt__", [](hpxpy::ndarray const& a, py::object const& b) {
            return hpxpy::ops::dispatch_compare(a, b, hpxpy::ops::Gt{});
        })
        .def("__ge__", [](hpxpy::ndarray const& a, py::object const& b) {
            return hpxpy::ops::dispatch_compare(a, b, hpxpy::ops::Ge{});
        });

    // Array creation functions
    m.def("_zeros", &hpxpy::zeros,
        py::arg("shape"), py::arg("dtype"),
        "Create a zero-filled array.");

    m.def("_ones", &hpxpy::ones,
        py::arg("shape"), py::arg("dtype"),
        "Create a one-filled array.");

    m.def("_empty", &hpxpy::empty,
        py::arg("shape"), py::arg("dtype"),
        "Create an uninitialized array.");

    m.def("_arange", &hpxpy::arange,
        py::arg("start"), py::arg("stop"), py::arg("step"), py::arg("dtype"),
        "Create an array with evenly spaced values.");

    m.def("_array_from_numpy", &hpxpy::array_from_numpy,
        py::arg("arr"), py::arg("copy"),
        "Create an array from a NumPy array.");
}
