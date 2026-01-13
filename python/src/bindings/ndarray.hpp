// HPXPy - ndarray class definition
//
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <numeric>
#include <sstream>
#include <vector>

namespace py = pybind11;

namespace hpxpy {

// HPXPy ndarray class
// For Phase 1, this is a thin wrapper around a contiguous buffer
// Later phases will add distributed storage via partitioned_vector
class ndarray {
public:
    // Construct from shape and dtype
    ndarray(std::vector<py::ssize_t> shape, py::dtype dtype)
        : shape_(std::move(shape))
        , dtype_(dtype)
        , strides_(compute_strides(shape_, dtype_.itemsize()))
    {
        size_ = compute_size(shape_);
        data_.resize(size_ * dtype_.itemsize());
    }

    // Construct from NumPy array (with optional copy)
    ndarray(py::array np_array, bool /*copy*/)
        : dtype_(np_array.dtype())
    {
        // Get shape
        for (py::ssize_t i = 0; i < np_array.ndim(); ++i) {
            shape_.push_back(np_array.shape(i));
        }
        size_ = compute_size(shape_);
        strides_ = compute_strides(shape_, dtype_.itemsize());

        // Allocate and copy data
        size_t nbytes = size_ * dtype_.itemsize();
        data_.resize(nbytes);

        // Make array contiguous if needed
        py::array contiguous = py::array::ensure(np_array,
            py::array::c_style | py::array::forcecast);

        std::memcpy(data_.data(), contiguous.data(), nbytes);
    }

    // Properties
    std::vector<py::ssize_t> const& shape() const { return shape_; }
    py::dtype dtype() const { return dtype_; }
    py::ssize_t size() const { return size_; }
    py::ssize_t ndim() const { return static_cast<py::ssize_t>(shape_.size()); }
    std::vector<py::ssize_t> const& strides() const { return strides_; }

    // Get raw data pointer
    void* data() { return data_.data(); }
    void const* data() const { return data_.data(); }
    size_t nbytes() const { return data_.size(); }

    // Convert to NumPy array
    py::array to_numpy() const {
        // Create NumPy array that copies our data
        return py::array(dtype_, shape_, strides_, data_.data());
    }

    // Get typed data pointer (for algorithm implementations)
    template<typename T>
    T* typed_data() {
        return reinterpret_cast<T*>(data_.data());
    }

    template<typename T>
    T const* typed_data() const {
        return reinterpret_cast<T const*>(data_.data());
    }

private:
    static std::vector<py::ssize_t> compute_strides(
        std::vector<py::ssize_t> const& shape,
        py::ssize_t itemsize)
    {
        std::vector<py::ssize_t> strides(shape.size());
        if (shape.empty()) return strides;

        strides.back() = itemsize;
        for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        return strides;
    }

    static py::ssize_t compute_size(std::vector<py::ssize_t> const& shape) {
        if (shape.empty()) return 0;
        return std::accumulate(shape.begin(), shape.end(),
            py::ssize_t{1}, std::multiplies<py::ssize_t>{});
    }

    std::vector<py::ssize_t> shape_;
    py::dtype dtype_;
    std::vector<py::ssize_t> strides_;
    py::ssize_t size_;
    std::vector<char> data_;  // Raw byte storage
};

}  // namespace hpxpy
