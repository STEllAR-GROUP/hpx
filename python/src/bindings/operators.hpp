// HPXPy - Operator implementations
//
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include "ndarray.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <hpx/algorithm.hpp>
#include <hpx/execution.hpp>

#include <cmath>
#include <functional>
#include <memory>
#include <stdexcept>

namespace py = pybind11;

namespace hpxpy {
namespace ops {

// Helper to check if shapes are compatible for broadcasting
inline bool shapes_equal(std::vector<py::ssize_t> const& a, std::vector<py::ssize_t> const& b) {
    return a == b;
}

// Apply binary operation element-wise (same shape arrays)
template<typename T, typename Op>
std::shared_ptr<ndarray> binary_op(ndarray const& a, ndarray const& b, Op op) {
    if (!shapes_equal(a.shape(), b.shape())) {
        throw std::runtime_error("Shapes must match for element-wise operation");
    }

    auto result = std::make_shared<ndarray>(a.shape(), a.dtype());

    T const* a_ptr = a.typed_data<T>();
    T const* b_ptr = b.typed_data<T>();
    T* r_ptr = result->typed_data<T>();
    py::ssize_t n = a.size();

    // Use HPX parallel for_each
    py::gil_scoped_release release;
    hpx::for_each(hpx::execution::par, r_ptr, r_ptr + n,
        [a_ptr, b_ptr, r_ptr, op](T& r) {
            auto idx = &r - r_ptr;
            r = op(a_ptr[idx], b_ptr[idx]);
        });

    return result;
}

// Apply binary operation with scalar (array op scalar)
template<typename T, typename Op>
std::shared_ptr<ndarray> binary_op_scalar(ndarray const& a, T scalar, Op op) {
    auto result = std::make_shared<ndarray>(a.shape(), a.dtype());

    T const* a_ptr = a.typed_data<T>();
    T* r_ptr = result->typed_data<T>();
    py::ssize_t n = a.size();

    py::gil_scoped_release release;
    hpx::for_each(hpx::execution::par, r_ptr, r_ptr + n,
        [a_ptr, r_ptr, scalar, op](T& r) {
            auto idx = &r - r_ptr;
            r = op(a_ptr[idx], scalar);
        });

    return result;
}

// Apply binary operation with scalar (scalar op array)
template<typename T, typename Op>
std::shared_ptr<ndarray> scalar_binary_op(T scalar, ndarray const& a, Op op) {
    auto result = std::make_shared<ndarray>(a.shape(), a.dtype());

    T const* a_ptr = a.typed_data<T>();
    T* r_ptr = result->typed_data<T>();
    py::ssize_t n = a.size();

    py::gil_scoped_release release;
    hpx::for_each(hpx::execution::par, r_ptr, r_ptr + n,
        [a_ptr, r_ptr, scalar, op](T& r) {
            auto idx = &r - r_ptr;
            r = op(scalar, a_ptr[idx]);
        });

    return result;
}

// Apply unary operation element-wise
template<typename T, typename Op>
std::shared_ptr<ndarray> unary_op(ndarray const& a, Op op) {
    auto result = std::make_shared<ndarray>(a.shape(), a.dtype());

    T const* a_ptr = a.typed_data<T>();
    T* r_ptr = result->typed_data<T>();
    py::ssize_t n = a.size();

    py::gil_scoped_release release;
    hpx::for_each(hpx::execution::par, r_ptr, r_ptr + n,
        [a_ptr, r_ptr, op](T& r) {
            auto idx = &r - r_ptr;
            r = op(a_ptr[idx]);
        });

    return result;
}

// Apply unary operation element-wise with output dtype
template<typename TIn, typename TOut, typename Op>
std::shared_ptr<ndarray> unary_op_out(ndarray const& a, py::dtype out_dtype, Op op) {
    auto result = std::make_shared<ndarray>(a.shape(), out_dtype);

    TIn const* a_ptr = a.typed_data<TIn>();
    TOut* r_ptr = result->typed_data<TOut>();
    py::ssize_t n = a.size();

    py::gil_scoped_release release;
    hpx::for_each(hpx::execution::par, r_ptr, r_ptr + n,
        [a_ptr, r_ptr, op](TOut& r) {
            auto idx = &r - r_ptr;
            r = op(a_ptr[idx]);
        });

    return result;
}

// Dispatch binary operation based on dtype
template<typename Op>
std::shared_ptr<ndarray> dispatch_binary(ndarray const& a, ndarray const& b, Op op) {
    char kind = a.dtype().kind();
    int itemsize = a.dtype().itemsize();

    if (kind == 'f') {
        if (itemsize == 8) return binary_op<double>(a, b, op);
        if (itemsize == 4) return binary_op<float>(a, b, op);
    } else if (kind == 'i') {
        if (itemsize == 8) return binary_op<int64_t>(a, b, op);
        if (itemsize == 4) return binary_op<int32_t>(a, b, op);
    } else if (kind == 'u') {
        if (itemsize == 8) return binary_op<uint64_t>(a, b, op);
        if (itemsize == 4) return binary_op<uint32_t>(a, b, op);
    }
    throw std::runtime_error("Unsupported dtype for operation");
}

// Dispatch binary operation with Python object (could be scalar or array)
template<typename Op>
std::shared_ptr<ndarray> dispatch_binary_py(ndarray const& a, py::object const& b, Op op) {
    // Check if b is an ndarray
    if (py::isinstance<ndarray>(b)) {
        return dispatch_binary(a, b.cast<ndarray const&>(), op);
    }

    // Otherwise treat as scalar
    char kind = a.dtype().kind();
    int itemsize = a.dtype().itemsize();

    if (kind == 'f') {
        double scalar = b.cast<double>();
        if (itemsize == 8) return binary_op_scalar<double>(a, scalar, op);
        if (itemsize == 4) return binary_op_scalar<float>(a, static_cast<float>(scalar), op);
    } else if (kind == 'i') {
        int64_t scalar = b.cast<int64_t>();
        if (itemsize == 8) return binary_op_scalar<int64_t>(a, scalar, op);
        if (itemsize == 4) return binary_op_scalar<int32_t>(a, static_cast<int32_t>(scalar), op);
    } else if (kind == 'u') {
        uint64_t scalar = b.cast<uint64_t>();
        if (itemsize == 8) return binary_op_scalar<uint64_t>(a, scalar, op);
        if (itemsize == 4) return binary_op_scalar<uint32_t>(a, static_cast<uint32_t>(scalar), op);
    }
    throw std::runtime_error("Unsupported dtype for operation");
}

// Dispatch reverse binary operation (scalar op array)
template<typename Op>
std::shared_ptr<ndarray> dispatch_rbinary_py(ndarray const& a, py::object const& b, Op op) {
    char kind = a.dtype().kind();
    int itemsize = a.dtype().itemsize();

    if (kind == 'f') {
        double scalar = b.cast<double>();
        if (itemsize == 8) return scalar_binary_op<double>(scalar, a, op);
        if (itemsize == 4) return scalar_binary_op<float>(static_cast<float>(scalar), a, op);
    } else if (kind == 'i') {
        int64_t scalar = b.cast<int64_t>();
        if (itemsize == 8) return scalar_binary_op<int64_t>(scalar, a, op);
        if (itemsize == 4) return scalar_binary_op<int32_t>(static_cast<int32_t>(scalar), a, op);
    }
    throw std::runtime_error("Unsupported dtype for operation");
}

// Dispatch unary operation
template<typename Op>
std::shared_ptr<ndarray> dispatch_unary(ndarray const& a, Op op) {
    char kind = a.dtype().kind();
    int itemsize = a.dtype().itemsize();

    if (kind == 'f') {
        if (itemsize == 8) return unary_op<double>(a, op);
        if (itemsize == 4) return unary_op<float>(a, op);
    } else if (kind == 'i') {
        if (itemsize == 8) return unary_op<int64_t>(a, op);
        if (itemsize == 4) return unary_op<int32_t>(a, op);
    }
    throw std::runtime_error("Unsupported dtype for operation");
}

// Comparison operations return bool array
template<typename T, typename Op>
std::shared_ptr<ndarray> compare_op(ndarray const& a, ndarray const& b, Op op) {
    if (!shapes_equal(a.shape(), b.shape())) {
        throw std::runtime_error("Shapes must match for comparison");
    }

    auto result = std::make_shared<ndarray>(a.shape(), py::dtype::of<bool>());

    T const* a_ptr = a.typed_data<T>();
    T const* b_ptr = b.typed_data<T>();
    bool* r_ptr = result->typed_data<bool>();
    py::ssize_t n = a.size();

    py::gil_scoped_release release;
    hpx::for_each(hpx::execution::par, r_ptr, r_ptr + n,
        [a_ptr, b_ptr, r_ptr, op](bool& r) {
            auto idx = &r - r_ptr;
            r = op(a_ptr[idx], b_ptr[idx]);
        });

    return result;
}

template<typename T, typename Op>
std::shared_ptr<ndarray> compare_op_scalar(ndarray const& a, T scalar, Op op) {
    auto result = std::make_shared<ndarray>(a.shape(), py::dtype::of<bool>());

    T const* a_ptr = a.typed_data<T>();
    bool* r_ptr = result->typed_data<bool>();
    py::ssize_t n = a.size();

    py::gil_scoped_release release;
    hpx::for_each(hpx::execution::par, r_ptr, r_ptr + n,
        [a_ptr, r_ptr, scalar, op](bool& r) {
            auto idx = &r - r_ptr;
            r = op(a_ptr[idx], scalar);
        });

    return result;
}

template<typename Op>
std::shared_ptr<ndarray> dispatch_compare(ndarray const& a, py::object const& b, Op op) {
    char kind = a.dtype().kind();
    int itemsize = a.dtype().itemsize();

    if (py::isinstance<ndarray>(b)) {
        ndarray const& b_arr = b.cast<ndarray const&>();
        if (kind == 'f') {
            if (itemsize == 8) return compare_op<double>(a, b_arr, op);
            if (itemsize == 4) return compare_op<float>(a, b_arr, op);
        } else if (kind == 'i') {
            if (itemsize == 8) return compare_op<int64_t>(a, b_arr, op);
            if (itemsize == 4) return compare_op<int32_t>(a, b_arr, op);
        }
    } else {
        // Scalar comparison
        if (kind == 'f') {
            double scalar = b.cast<double>();
            if (itemsize == 8) return compare_op_scalar<double>(a, scalar, op);
            if (itemsize == 4) return compare_op_scalar<float>(a, static_cast<float>(scalar), op);
        } else if (kind == 'i') {
            int64_t scalar = b.cast<int64_t>();
            if (itemsize == 8) return compare_op_scalar<int64_t>(a, scalar, op);
            if (itemsize == 4) return compare_op_scalar<int32_t>(a, static_cast<int32_t>(scalar), op);
        }
    }
    throw std::runtime_error("Unsupported dtype for comparison");
}

// Arithmetic operations
struct Add { template<typename T> T operator()(T a, T b) const { return a + b; } };
struct Sub { template<typename T> T operator()(T a, T b) const { return a - b; } };
struct Mul { template<typename T> T operator()(T a, T b) const { return a * b; } };
struct Div { template<typename T> T operator()(T a, T b) const { return a / b; } };
struct Mod { template<typename T> T operator()(T a, T b) const { return std::fmod(static_cast<double>(a), static_cast<double>(b)); } };
struct Pow { template<typename T> T operator()(T a, T b) const { return std::pow(a, b); } };
struct FloorDiv {
    template<typename T>
    T operator()(T a, T b) const {
        return static_cast<T>(std::floor(static_cast<double>(a) / static_cast<double>(b)));
    }
};

// Comparison operations
struct Eq { template<typename T> bool operator()(T a, T b) const { return a == b; } };
struct Ne { template<typename T> bool operator()(T a, T b) const { return a != b; } };
struct Lt { template<typename T> bool operator()(T a, T b) const { return a < b; } };
struct Le { template<typename T> bool operator()(T a, T b) const { return a <= b; } };
struct Gt { template<typename T> bool operator()(T a, T b) const { return a > b; } };
struct Ge { template<typename T> bool operator()(T a, T b) const { return a >= b; } };

// Unary operations
struct Neg { template<typename T> T operator()(T a) const { return -a; } };
struct Pos { template<typename T> T operator()(T a) const { return +a; } };
struct Abs { template<typename T> T operator()(T a) const { return std::abs(a); } };

// Math functions
struct Sqrt { template<typename T> T operator()(T a) const { return std::sqrt(a); } };
struct Square { template<typename T> T operator()(T a) const { return a * a; } };
struct Exp { template<typename T> T operator()(T a) const { return std::exp(a); } };
struct Exp2 { template<typename T> T operator()(T a) const { return std::exp2(a); } };
struct Log { template<typename T> T operator()(T a) const { return std::log(a); } };
struct Log2 { template<typename T> T operator()(T a) const { return std::log2(a); } };
struct Log10 { template<typename T> T operator()(T a) const { return std::log10(a); } };
struct Sin { template<typename T> T operator()(T a) const { return std::sin(a); } };
struct Cos { template<typename T> T operator()(T a) const { return std::cos(a); } };
struct Tan { template<typename T> T operator()(T a) const { return std::tan(a); } };
struct Asin { template<typename T> T operator()(T a) const { return std::asin(a); } };
struct Acos { template<typename T> T operator()(T a) const { return std::acos(a); } };
struct Atan { template<typename T> T operator()(T a) const { return std::atan(a); } };
struct Sinh { template<typename T> T operator()(T a) const { return std::sinh(a); } };
struct Cosh { template<typename T> T operator()(T a) const { return std::cosh(a); } };
struct Tanh { template<typename T> T operator()(T a) const { return std::tanh(a); } };
struct Floor { template<typename T> T operator()(T a) const { return std::floor(a); } };
struct Ceil { template<typename T> T operator()(T a) const { return std::ceil(a); } };
struct Trunc { template<typename T> T operator()(T a) const { return std::trunc(a); } };
struct Sign {
    template<typename T>
    T operator()(T a) const {
        return (T(0) < a) - (a < T(0));
    }
};

}  // namespace ops
}  // namespace hpxpy
