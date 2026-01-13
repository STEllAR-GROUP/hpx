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
    ndarray(py::array np_array, bool copy)
        : dtype_(np_array.dtype())
        , np_base_(copy ? py::none() : py::reinterpret_borrow<py::object>(np_array))
    {
        // Get shape
        for (py::ssize_t i = 0; i < np_array.ndim(); ++i) {
            shape_.push_back(np_array.shape(i));
        }
        size_ = compute_size(shape_);

        // Make array contiguous if needed
        py::array contiguous = py::array::ensure(np_array,
            py::array::c_style | py::array::forcecast);

        if (copy || contiguous.ptr() != np_array.ptr()) {
            // Need to copy: either requested or array wasn't contiguous
            strides_ = compute_strides(shape_, dtype_.itemsize());
            size_t nbytes = size_ * dtype_.itemsize();
            data_.resize(nbytes);
            std::memcpy(data_.data(), contiguous.data(), nbytes);
            external_data_ = nullptr;
            np_base_ = py::none();
        } else {
            // Zero-copy: use numpy's buffer directly
            // Keep reference to numpy array to ensure lifetime
            strides_ = compute_strides(shape_, dtype_.itemsize());
            external_data_ = const_cast<void*>(np_array.data());
            np_base_ = py::reinterpret_borrow<py::object>(np_array);
        }
    }

    // Properties
    std::vector<py::ssize_t> const& shape() const { return shape_; }
    py::dtype dtype() const { return dtype_; }
    py::ssize_t size() const { return size_; }
    py::ssize_t ndim() const { return static_cast<py::ssize_t>(shape_.size()); }
    std::vector<py::ssize_t> const& strides() const { return strides_; }

    // Get raw data pointer
    void* data() {
        return external_data_ ? external_data_ : data_.data();
    }
    void const* data() const {
        return external_data_ ? external_data_ : data_.data();
    }
    size_t nbytes() const {
        return external_data_ ? size_ * dtype_.itemsize() : data_.size();
    }

    // Check if this is a view (references external data)
    bool is_view() const { return external_data_ != nullptr; }

    // Convert to NumPy array (zero-copy view when possible)
    py::array to_numpy() const {
        if (external_data_) {
            // Return a view of the external data, keeping the original numpy array alive
            return py::array(dtype_, shape_, strides_, external_data_, np_base_);
        } else {
            // We own the data - create a copy for safety
            // (the ndarray could be destroyed while numpy array is still in use)
            return py::array(dtype_, shape_, strides_, data_.data());
        }
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
    std::vector<char> data_;      // Raw byte storage (when we own data)
    void* external_data_ = nullptr;  // Pointer to external data (zero-copy view)
    py::object np_base_;          // Keep numpy array alive for zero-copy views
};

}  // namespace hpxpy
