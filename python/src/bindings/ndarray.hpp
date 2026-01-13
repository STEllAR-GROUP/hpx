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
class ndarray : public std::enable_shared_from_this<ndarray> {
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
            if (data_owner_) {
                // This is a slice view - we need to return a contiguous copy
                // because NumPy can't keep an arbitrary C++ shared_ptr alive.
                // The slice may have non-contiguous strides from step slicing.
                py::array result(dtype_, shape_);
                copy_to_contiguous(result.mutable_data(), shape_, strides_,
                    external_data_, dtype_.itemsize());
                return result;
            } else {
                // Return a view of the external data, keeping the original numpy array alive
                return py::array(dtype_, shape_, strides_, external_data_, np_base_);
            }
        } else {
            // We own the data - create a copy for safety
            // (the ndarray could be destroyed while numpy array is still in use)
            return py::array(dtype_, shape_, strides_, data_.data());
        }
    }

    // Get typed data pointer (for algorithm implementations)
    template<typename T>
    T* typed_data() {
        return reinterpret_cast<T*>(data());
    }

    template<typename T>
    T const* typed_data() const {
        return reinterpret_cast<T const*>(data());
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

    // Copy strided data to contiguous buffer
    static void copy_to_contiguous(
        void* dst,
        std::vector<py::ssize_t> const& shape,
        std::vector<py::ssize_t> const& strides,
        void const* src,
        py::ssize_t itemsize)
    {
        if (shape.empty()) return;

        char* dst_ptr = static_cast<char*>(dst);
        char const* src_ptr = static_cast<char const*>(src);

        if (shape.size() == 1) {
            // 1D case: iterate with stride
            for (py::ssize_t i = 0; i < shape[0]; ++i) {
                std::memcpy(dst_ptr, src_ptr, itemsize);
                dst_ptr += itemsize;
                src_ptr += strides[0];
            }
        } else {
            // Multi-dimensional: recursive copy
            py::ssize_t inner_size = 1;
            for (size_t i = 1; i < shape.size(); ++i) {
                inner_size *= shape[i];
            }

            std::vector<py::ssize_t> inner_shape(shape.begin() + 1, shape.end());
            std::vector<py::ssize_t> inner_strides(strides.begin() + 1, strides.end());

            for (py::ssize_t i = 0; i < shape[0]; ++i) {
                copy_to_contiguous(dst_ptr, inner_shape, inner_strides,
                    src_ptr, itemsize);
                dst_ptr += inner_size * itemsize;
                src_ptr += strides[0];
            }
        }
    }

    std::vector<py::ssize_t> shape_;
    py::dtype dtype_;
    std::vector<py::ssize_t> strides_;
    py::ssize_t size_;
    std::vector<char> data_;      // Raw byte storage (when we own data)
    void* external_data_ = nullptr;  // Pointer to external data (zero-copy view)
    py::object np_base_;          // Keep numpy array alive for zero-copy views
    std::shared_ptr<const ndarray> data_owner_;  // Keep parent alive for slice views

public:
    // Tag type for slice construction (public for make_shared, but tag prevents misuse)
    struct slice_view_tag {};

    // Constructor for slice views (uses tag to prevent accidental use)
    ndarray(slice_view_tag,
            std::vector<py::ssize_t> shape,
            std::vector<py::ssize_t> strides,
            py::dtype dtype,
            void* data_ptr,
            std::shared_ptr<const ndarray> owner)
        : shape_(std::move(shape))
        , dtype_(dtype)
        , strides_(std::move(strides))
        , size_(compute_size(shape_))
        , external_data_(data_ptr)
        , data_owner_(std::move(owner))
    {}
    // Check if array data is contiguous (C-order)
    bool is_contiguous() const {
        if (shape_.empty()) return true;

        py::ssize_t expected_stride = dtype_.itemsize();
        for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i) {
            if (strides_[i] != expected_stride) {
                return false;
            }
            expected_stride *= shape_[i];
        }
        return true;
    }

    // Reshape array to new shape (returns view if contiguous, copy otherwise)
    std::shared_ptr<ndarray> reshape(std::vector<py::ssize_t> new_shape) const {
        // Compute new size and handle -1 dimension
        py::ssize_t new_size = 1;
        int infer_dim = -1;
        for (size_t i = 0; i < new_shape.size(); ++i) {
            if (new_shape[i] == -1) {
                if (infer_dim >= 0) {
                    throw std::runtime_error("Can only specify one unknown dimension (-1)");
                }
                infer_dim = static_cast<int>(i);
            } else if (new_shape[i] < 0) {
                throw std::runtime_error("Invalid shape dimension: negative value");
            } else {
                new_size *= new_shape[i];
            }
        }

        // Infer the -1 dimension if present
        if (infer_dim >= 0) {
            if (new_size == 0) {
                throw std::runtime_error("Cannot infer dimension with zero-sized dimensions");
            }
            if (size_ % new_size != 0) {
                throw std::runtime_error("Cannot reshape array: size mismatch");
            }
            new_shape[infer_dim] = size_ / new_size;
            new_size = size_;
        }

        // Verify size matches
        if (new_size != size_) {
            std::ostringstream ss;
            ss << "Cannot reshape array of size " << size_ << " into shape (";
            for (size_t i = 0; i < new_shape.size(); ++i) {
                if (i > 0) ss << ", ";
                ss << new_shape[i];
            }
            ss << ")";
            throw std::runtime_error(ss.str());
        }

        // Compute new strides for C-contiguous layout
        std::vector<py::ssize_t> new_strides = compute_strides(new_shape, dtype_.itemsize());

        if (is_contiguous()) {
            // Return a view with new shape
            return std::make_shared<ndarray>(
                slice_view_tag{},
                std::move(new_shape),
                std::move(new_strides),
                dtype_,
                const_cast<void*>(data()),
                shared_from_this()
            );
        } else {
            // Non-contiguous: must copy to new contiguous array
            auto result = std::make_shared<ndarray>(new_shape, dtype_);
            copy_to_contiguous(result->data(), shape_, strides_, data(), dtype_.itemsize());
            return result;
        }
    }

    // Flatten array to 1D (always returns a copy)
    std::shared_ptr<ndarray> flatten() const {
        std::vector<py::ssize_t> flat_shape = {size_};
        auto result = std::make_shared<ndarray>(flat_shape, dtype_);

        if (is_contiguous()) {
            // Simple copy for contiguous arrays
            std::memcpy(result->data(), data(), size_ * dtype_.itemsize());
        } else {
            // Copy with stride handling for non-contiguous
            copy_to_contiguous(result->data(), shape_, strides_, data(), dtype_.itemsize());
        }
        return result;
    }

    // Ravel array to 1D (returns view if contiguous, copy otherwise)
    std::shared_ptr<ndarray> ravel() const {
        std::vector<py::ssize_t> flat_shape = {size_};

        if (is_contiguous()) {
            // Return a view
            std::vector<py::ssize_t> flat_strides = {dtype_.itemsize()};
            return std::make_shared<ndarray>(
                slice_view_tag{},
                std::move(flat_shape),
                std::move(flat_strides),
                dtype_,
                const_cast<void*>(data()),
                shared_from_this()
            );
        } else {
            // Non-contiguous: must copy (same as flatten)
            return flatten();
        }
    }

    // Slice a 1D array: arr[start:stop:step]
    std::shared_ptr<ndarray> getitem_slice(py::slice slice) const {
        if (shape_.empty()) {
            throw std::runtime_error("Cannot slice a 0-dimensional array");
        }

        // Compute slice parameters for first dimension
        py::ssize_t start, stop, step, length;
        if (!slice.compute(shape_[0], &start, &stop, &step, &length)) {
            throw py::error_already_set();
        }

        // Build new shape: first dimension is the slice length
        std::vector<py::ssize_t> new_shape = shape_;
        new_shape[0] = length;

        // Build new strides: first dimension stride is multiplied by step
        std::vector<py::ssize_t> new_strides = strides_;
        new_strides[0] = strides_[0] * step;

        // Compute data pointer offset
        char const* base_ptr = static_cast<char const*>(data());
        void* new_data = const_cast<char*>(base_ptr + start * strides_[0]);

        // Create slice view, keeping this array alive
        return std::make_shared<ndarray>(
            slice_view_tag{},
            std::move(new_shape),
            std::move(new_strides),
            dtype_,
            new_data,
            shared_from_this()
        );
    }
};

}  // namespace hpxpy
