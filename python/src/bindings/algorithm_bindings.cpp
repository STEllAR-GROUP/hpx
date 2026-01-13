// HPXPy - Algorithm bindings
//
// SPDX-License-Identifier: BSL-1.0

#include "ndarray.hpp"
#include "operators.hpp"

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
#include <random>
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

// Math functions using operators.hpp
std::shared_ptr<ndarray> sqrt_arr(std::shared_ptr<ndarray> arr) {
    return ops::dispatch_unary(*arr, ops::Sqrt{});
}

std::shared_ptr<ndarray> square_arr(std::shared_ptr<ndarray> arr) {
    return ops::dispatch_unary(*arr, ops::Square{});
}

std::shared_ptr<ndarray> exp_arr(std::shared_ptr<ndarray> arr) {
    return ops::dispatch_unary(*arr, ops::Exp{});
}

std::shared_ptr<ndarray> exp2_arr(std::shared_ptr<ndarray> arr) {
    return ops::dispatch_unary(*arr, ops::Exp2{});
}

std::shared_ptr<ndarray> log_arr(std::shared_ptr<ndarray> arr) {
    return ops::dispatch_unary(*arr, ops::Log{});
}

std::shared_ptr<ndarray> log2_arr(std::shared_ptr<ndarray> arr) {
    return ops::dispatch_unary(*arr, ops::Log2{});
}

std::shared_ptr<ndarray> log10_arr(std::shared_ptr<ndarray> arr) {
    return ops::dispatch_unary(*arr, ops::Log10{});
}

std::shared_ptr<ndarray> sin_arr(std::shared_ptr<ndarray> arr) {
    return ops::dispatch_unary(*arr, ops::Sin{});
}

std::shared_ptr<ndarray> cos_arr(std::shared_ptr<ndarray> arr) {
    return ops::dispatch_unary(*arr, ops::Cos{});
}

std::shared_ptr<ndarray> tan_arr(std::shared_ptr<ndarray> arr) {
    return ops::dispatch_unary(*arr, ops::Tan{});
}

std::shared_ptr<ndarray> arcsin_arr(std::shared_ptr<ndarray> arr) {
    return ops::dispatch_unary(*arr, ops::Asin{});
}

std::shared_ptr<ndarray> arccos_arr(std::shared_ptr<ndarray> arr) {
    return ops::dispatch_unary(*arr, ops::Acos{});
}

std::shared_ptr<ndarray> arctan_arr(std::shared_ptr<ndarray> arr) {
    return ops::dispatch_unary(*arr, ops::Atan{});
}

std::shared_ptr<ndarray> sinh_arr(std::shared_ptr<ndarray> arr) {
    return ops::dispatch_unary(*arr, ops::Sinh{});
}

std::shared_ptr<ndarray> cosh_arr(std::shared_ptr<ndarray> arr) {
    return ops::dispatch_unary(*arr, ops::Cosh{});
}

std::shared_ptr<ndarray> tanh_arr(std::shared_ptr<ndarray> arr) {
    return ops::dispatch_unary(*arr, ops::Tanh{});
}

std::shared_ptr<ndarray> floor_arr(std::shared_ptr<ndarray> arr) {
    return ops::dispatch_unary(*arr, ops::Floor{});
}

std::shared_ptr<ndarray> ceil_arr(std::shared_ptr<ndarray> arr) {
    return ops::dispatch_unary(*arr, ops::Ceil{});
}

std::shared_ptr<ndarray> trunc_arr(std::shared_ptr<ndarray> arr) {
    return ops::dispatch_unary(*arr, ops::Trunc{});
}

std::shared_ptr<ndarray> abs_arr(std::shared_ptr<ndarray> arr) {
    return ops::dispatch_unary(*arr, ops::Abs{});
}

std::shared_ptr<ndarray> sign_arr(std::shared_ptr<ndarray> arr) {
    return ops::dispatch_unary(*arr, ops::Sign{});
}

// Cumulative sum (scan)
std::shared_ptr<ndarray> cumsum(std::shared_ptr<ndarray> arr) {
    auto result = std::make_shared<ndarray>(arr->shape(), arr->dtype());

    char kind = arr->dtype().kind();
    auto itemsize = arr->dtype().itemsize();

    py::gil_scoped_release release;

    if (kind == 'f') {
        if (itemsize == 8) {
            auto const* src = arr->typed_data<double>();
            auto* dst = result->typed_data<double>();
            hpx::inclusive_scan(hpx::execution::par, src, src + arr->size(), dst);
        } else if (itemsize == 4) {
            auto const* src = arr->typed_data<float>();
            auto* dst = result->typed_data<float>();
            hpx::inclusive_scan(hpx::execution::par, src, src + arr->size(), dst);
        }
    } else if (kind == 'i') {
        if (itemsize == 8) {
            auto const* src = arr->typed_data<int64_t>();
            auto* dst = result->typed_data<int64_t>();
            hpx::inclusive_scan(hpx::execution::par, src, src + arr->size(), dst);
        } else if (itemsize == 4) {
            auto const* src = arr->typed_data<int32_t>();
            auto* dst = result->typed_data<int32_t>();
            hpx::inclusive_scan(hpx::execution::par, src, src + arr->size(), dst);
        }
    } else {
        py::gil_scoped_acquire acquire;
        throw std::runtime_error("Unsupported dtype for cumsum");
    }

    return result;
}

// Cumulative product (scan)
std::shared_ptr<ndarray> cumprod(std::shared_ptr<ndarray> arr) {
    auto result = std::make_shared<ndarray>(arr->shape(), arr->dtype());

    char kind = arr->dtype().kind();
    auto itemsize = arr->dtype().itemsize();

    py::gil_scoped_release release;

    if (kind == 'f') {
        if (itemsize == 8) {
            auto const* src = arr->typed_data<double>();
            auto* dst = result->typed_data<double>();
            hpx::inclusive_scan(hpx::execution::par, src, src + arr->size(), dst,
                std::multiplies<double>{});
        } else if (itemsize == 4) {
            auto const* src = arr->typed_data<float>();
            auto* dst = result->typed_data<float>();
            hpx::inclusive_scan(hpx::execution::par, src, src + arr->size(), dst,
                std::multiplies<float>{});
        }
    } else if (kind == 'i') {
        if (itemsize == 8) {
            auto const* src = arr->typed_data<int64_t>();
            auto* dst = result->typed_data<int64_t>();
            hpx::inclusive_scan(hpx::execution::par, src, src + arr->size(), dst,
                std::multiplies<int64_t>{});
        } else if (itemsize == 4) {
            auto const* src = arr->typed_data<int32_t>();
            auto* dst = result->typed_data<int32_t>();
            hpx::inclusive_scan(hpx::execution::par, src, src + arr->size(), dst,
                std::multiplies<int32_t>{});
        }
    } else {
        py::gil_scoped_acquire acquire;
        throw std::runtime_error("Unsupported dtype for cumprod");
    }

    return result;
}

// Random number generation
namespace random {

// Thread-local random engine
thread_local std::mt19937_64 rng{std::random_device{}()};

std::shared_ptr<ndarray> uniform(double low, double high, std::vector<py::ssize_t> shape) {
    auto result = std::make_shared<ndarray>(shape, py::dtype::of<double>());
    auto* data = result->typed_data<double>();
    py::ssize_t n = result->size();

    std::uniform_real_distribution<double> dist(low, high);

    // Generate random numbers (parallel generation with thread-local RNG)
    for (py::ssize_t i = 0; i < n; ++i) {
        data[i] = dist(rng);
    }

    return result;
}

std::shared_ptr<ndarray> randn(std::vector<py::ssize_t> shape) {
    auto result = std::make_shared<ndarray>(shape, py::dtype::of<double>());
    auto* data = result->typed_data<double>();
    py::ssize_t n = result->size();

    std::normal_distribution<double> dist(0.0, 1.0);

    for (py::ssize_t i = 0; i < n; ++i) {
        data[i] = dist(rng);
    }

    return result;
}

std::shared_ptr<ndarray> randint(int64_t low, int64_t high, std::vector<py::ssize_t> shape) {
    auto result = std::make_shared<ndarray>(shape, py::dtype::of<int64_t>());
    auto* data = result->typed_data<int64_t>();
    py::ssize_t n = result->size();

    std::uniform_int_distribution<int64_t> dist(low, high - 1);  // high is exclusive

    for (py::ssize_t i = 0; i < n; ++i) {
        data[i] = dist(rng);
    }

    return result;
}

std::shared_ptr<ndarray> rand(std::vector<py::ssize_t> shape) {
    return uniform(0.0, 1.0, shape);
}

void seed(uint64_t s) {
    rng.seed(s);
}

}  // namespace random

// Helper structs for operations (at namespace scope for template compatibility)
struct MaxOp {
    template<typename T> T operator()(T x, T y) const { return x > y ? x : y; }
};

struct MinOp {
    template<typename T> T operator()(T x, T y) const { return x < y ? x : y; }
};

struct ClipOp {
    double min_v, max_v;
    template<typename T> T operator()(T x) const {
        if (x < static_cast<T>(min_v)) return static_cast<T>(min_v);
        if (x > static_cast<T>(max_v)) return static_cast<T>(max_v);
        return x;
    }
};

// Element-wise maximum of two arrays
std::shared_ptr<ndarray> maximum(std::shared_ptr<ndarray> a, py::object b) {
    return ops::dispatch_binary_py(*a, b, MaxOp{});
}

// Element-wise minimum of two arrays
std::shared_ptr<ndarray> minimum(std::shared_ptr<ndarray> a, py::object b) {
    return ops::dispatch_binary_py(*a, b, MinOp{});
}

// Clip values to range
std::shared_ptr<ndarray> clip(std::shared_ptr<ndarray> arr, py::object a_min, py::object a_max) {
    double min_val = a_min.cast<double>();
    double max_val = a_max.cast<double>();
    return ops::dispatch_unary(*arr, ClipOp{min_val, max_val});
}

// Power function with scalar exponent
std::shared_ptr<ndarray> power(std::shared_ptr<ndarray> arr, py::object exponent) {
    return ops::dispatch_binary_py(*arr, exponent, ops::Pow{});
}

// where - conditional selection
std::shared_ptr<ndarray> where_arr(std::shared_ptr<ndarray> condition,
                                    std::shared_ptr<ndarray> x,
                                    std::shared_ptr<ndarray> y) {
    if (!ops::shapes_equal(condition->shape(), x->shape()) ||
        !ops::shapes_equal(condition->shape(), y->shape())) {
        throw std::runtime_error("Shapes must match for where()");
    }

    auto result = std::make_shared<ndarray>(x->shape(), x->dtype());

    char kind = x->dtype().kind();
    auto itemsize = x->dtype().itemsize();
    bool const* cond = condition->typed_data<bool>();

    py::gil_scoped_release release;

    if (kind == 'f') {
        if (itemsize == 8) {
            double const* x_ptr = x->typed_data<double>();
            double const* y_ptr = y->typed_data<double>();
            double* r_ptr = result->typed_data<double>();
            for (py::ssize_t i = 0; i < result->size(); ++i) {
                r_ptr[i] = cond[i] ? x_ptr[i] : y_ptr[i];
            }
        } else if (itemsize == 4) {
            float const* x_ptr = x->typed_data<float>();
            float const* y_ptr = y->typed_data<float>();
            float* r_ptr = result->typed_data<float>();
            for (py::ssize_t i = 0; i < result->size(); ++i) {
                r_ptr[i] = cond[i] ? x_ptr[i] : y_ptr[i];
            }
        }
    } else if (kind == 'i') {
        if (itemsize == 8) {
            int64_t const* x_ptr = x->typed_data<int64_t>();
            int64_t const* y_ptr = y->typed_data<int64_t>();
            int64_t* r_ptr = result->typed_data<int64_t>();
            for (py::ssize_t i = 0; i < result->size(); ++i) {
                r_ptr[i] = cond[i] ? x_ptr[i] : y_ptr[i];
            }
        } else if (itemsize == 4) {
            int32_t const* x_ptr = x->typed_data<int32_t>();
            int32_t const* y_ptr = y->typed_data<int32_t>();
            int32_t* r_ptr = result->typed_data<int32_t>();
            for (py::ssize_t i = 0; i < result->size(); ++i) {
                r_ptr[i] = cond[i] ? x_ptr[i] : y_ptr[i];
            }
        }
    } else {
        py::gil_scoped_acquire acquire;
        throw std::runtime_error("Unsupported dtype for where");
    }

    return result;
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

    // Math functions
    m.def("_sqrt", &hpxpy::sqrt_arr, py::arg("arr"), "Element-wise square root.");
    m.def("_square", &hpxpy::square_arr, py::arg("arr"), "Element-wise square.");
    m.def("_exp", &hpxpy::exp_arr, py::arg("arr"), "Element-wise exponential.");
    m.def("_exp2", &hpxpy::exp2_arr, py::arg("arr"), "Element-wise 2**x.");
    m.def("_log", &hpxpy::log_arr, py::arg("arr"), "Element-wise natural logarithm.");
    m.def("_log2", &hpxpy::log2_arr, py::arg("arr"), "Element-wise base-2 logarithm.");
    m.def("_log10", &hpxpy::log10_arr, py::arg("arr"), "Element-wise base-10 logarithm.");
    m.def("_sin", &hpxpy::sin_arr, py::arg("arr"), "Element-wise sine.");
    m.def("_cos", &hpxpy::cos_arr, py::arg("arr"), "Element-wise cosine.");
    m.def("_tan", &hpxpy::tan_arr, py::arg("arr"), "Element-wise tangent.");
    m.def("_arcsin", &hpxpy::arcsin_arr, py::arg("arr"), "Element-wise inverse sine.");
    m.def("_arccos", &hpxpy::arccos_arr, py::arg("arr"), "Element-wise inverse cosine.");
    m.def("_arctan", &hpxpy::arctan_arr, py::arg("arr"), "Element-wise inverse tangent.");
    m.def("_sinh", &hpxpy::sinh_arr, py::arg("arr"), "Element-wise hyperbolic sine.");
    m.def("_cosh", &hpxpy::cosh_arr, py::arg("arr"), "Element-wise hyperbolic cosine.");
    m.def("_tanh", &hpxpy::tanh_arr, py::arg("arr"), "Element-wise hyperbolic tangent.");
    m.def("_floor", &hpxpy::floor_arr, py::arg("arr"), "Element-wise floor.");
    m.def("_ceil", &hpxpy::ceil_arr, py::arg("arr"), "Element-wise ceiling.");
    m.def("_trunc", &hpxpy::trunc_arr, py::arg("arr"), "Element-wise truncation.");
    m.def("_abs", &hpxpy::abs_arr, py::arg("arr"), "Element-wise absolute value.");
    m.def("_sign", &hpxpy::sign_arr, py::arg("arr"), "Element-wise sign.");

    // Scan operations
    m.def("_cumsum", &hpxpy::cumsum, py::arg("arr"), "Cumulative sum.");
    m.def("_cumprod", &hpxpy::cumprod, py::arg("arr"), "Cumulative product.");

    // Additional functions
    m.def("_maximum", &hpxpy::maximum, py::arg("a"), py::arg("b"),
        "Element-wise maximum of two arrays.");
    m.def("_minimum", &hpxpy::minimum, py::arg("a"), py::arg("b"),
        "Element-wise minimum of two arrays.");
    m.def("_clip", &hpxpy::clip, py::arg("arr"), py::arg("a_min"), py::arg("a_max"),
        "Clip values to a range.");
    m.def("_power", &hpxpy::power, py::arg("arr"), py::arg("exponent"),
        "Element-wise power.");
    m.def("_where", &hpxpy::where_arr, py::arg("condition"), py::arg("x"), py::arg("y"),
        "Return elements chosen from x or y depending on condition.");

    // Random submodule
    auto random = m.def_submodule("random", "Random number generation.");
    random.def("_uniform", &hpxpy::random::uniform,
        py::arg("low"), py::arg("high"), py::arg("shape"),
        "Uniform distribution over [low, high).");
    random.def("_randn", &hpxpy::random::randn, py::arg("shape"),
        "Standard normal distribution.");
    random.def("_randint", &hpxpy::random::randint,
        py::arg("low"), py::arg("high"), py::arg("shape"),
        "Random integers from [low, high).");
    random.def("_rand", &hpxpy::random::rand, py::arg("shape"),
        "Uniform distribution over [0, 1).");
    random.def("seed", &hpxpy::random::seed, py::arg("seed"),
        "Seed the random number generator.");
}

void bind_execution(py::module_& m) {
    // Create execution submodule
    auto exec = m.def_submodule("execution",
        "Execution policies for controlling parallel execution.");

    // Phase 2: Full execution policy support
    py::class_<hpx::execution::sequenced_policy>(exec, "sequenced_policy",
        "Sequential execution policy.")
        .def(py::init<>());

    py::class_<hpx::execution::parallel_policy>(exec, "parallel_policy",
        "Parallel execution policy.")
        .def(py::init<>());

    py::class_<hpx::execution::parallel_unsequenced_policy>(exec, "parallel_unsequenced_policy",
        "Parallel unsequenced execution policy.")
        .def(py::init<>());

    exec.attr("seq") = hpx::execution::seq;
    exec.attr("par") = hpx::execution::par;
    exec.attr("par_unseq") = hpx::execution::par_unseq;
}
