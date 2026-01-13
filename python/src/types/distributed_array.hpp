// HPXPy - Distribution utilities
//
// SPDX-License-Identifier: BSL-1.0
//
// This header provides distribution policy definitions and utilities
// for distributed array support. In single-locality mode, arrays are
// stored locally with distribution metadata. In multi-locality mode,
// arrays would use HPX partitioned_vector for actual distribution.

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <hpx/hpx.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>

namespace py = pybind11;

namespace hpxpy {

// Distribution policy types
enum class DistributionPolicy {
    None,       // Local array (no distribution)
    Block,      // Block distribution (contiguous chunks)
    Cyclic      // Cyclic distribution (round-robin)
};

// Convert string to distribution policy
inline DistributionPolicy parse_distribution(py::object const& dist) {
    if (dist.is_none()) {
        return DistributionPolicy::None;
    }

    if (py::isinstance<py::str>(dist)) {
        std::string s = dist.cast<std::string>();
        if (s == "block") return DistributionPolicy::Block;
        if (s == "cyclic") return DistributionPolicy::Cyclic;
        if (s == "none" || s == "local") return DistributionPolicy::None;
        throw std::invalid_argument("Unknown distribution policy: " + s);
    }

    // Could be an enum value
    if (py::isinstance<DistributionPolicy>(dist)) {
        return dist.cast<DistributionPolicy>();
    }

    throw std::invalid_argument("Invalid distribution type");
}

// Distribution info that can be attached to arrays
struct DistributionInfo {
    DistributionPolicy policy = DistributionPolicy::None;
    std::size_t num_partitions = 1;
    std::size_t chunk_size = 0;  // 0 means automatic
    std::uint32_t locality_id = 0;

    bool is_distributed() const {
        return policy != DistributionPolicy::None && num_partitions > 1;
    }
};

// Get current locality ID (for introspection)
inline std::uint32_t get_current_locality() {
    return hpx::get_locality_id();
}

// Get number of localities
inline std::size_t get_num_localities() {
    return hpx::get_num_localities(hpx::launch::sync);
}

// Distributed array class
// In single-locality mode, this stores data locally with distribution metadata.
// The API is designed to work seamlessly when extended to multi-locality.
template<typename T>
class distributed_array {
public:
    using value_type = T;

    // Default constructor
    distributed_array()
        : size_(0)
        , num_partitions_(1)
        , locality_id_(0)
        , policy_(DistributionPolicy::None)
    {}

    // Create with shape and distribution
    distributed_array(
        std::vector<py::ssize_t> shape,
        DistributionPolicy policy = DistributionPolicy::None,
        std::size_t num_partitions = 0)
        : shape_(std::move(shape))
        , policy_(policy)
    {
        // Compute total size
        size_ = 1;
        for (auto dim : shape_) {
            size_ *= dim;
        }

        // Determine number of partitions
        std::size_t num_locs = get_num_localities();
        if (num_partitions == 0) {
            // Default: one partition per locality
            num_partitions_ = (policy == DistributionPolicy::None) ? 1 : num_locs;
        } else {
            num_partitions_ = num_partitions;
        }

        // Allocate local storage
        data_.resize(static_cast<std::size_t>(size_));

        locality_id_ = get_current_locality();
    }

    // Create from local data
    distributed_array(
        std::vector<T> const& local_data,
        std::vector<py::ssize_t> shape,
        DistributionPolicy policy = DistributionPolicy::None)
        : shape_(std::move(shape))
        , policy_(policy)
        , size_(static_cast<py::ssize_t>(local_data.size()))
        , num_partitions_(1)
        , locality_id_(get_current_locality())
        , data_(local_data)
    {
    }

    // Properties
    std::vector<py::ssize_t> const& shape() const { return shape_; }
    py::ssize_t size() const { return size_; }
    py::ssize_t ndim() const { return static_cast<py::ssize_t>(shape_.size()); }
    DistributionPolicy policy() const { return policy_; }
    std::size_t num_partitions() const { return num_partitions_; }
    std::uint32_t locality_id() const { return locality_id_; }

    bool is_distributed() const {
        return policy_ != DistributionPolicy::None &&
               num_partitions_ > 1 &&
               get_num_localities() > 1;
    }

    // Get partition info
    DistributionInfo get_distribution_info() const {
        std::size_t chunk = (num_partitions_ > 0) ?
            static_cast<std::size_t>(size_) / num_partitions_ : static_cast<std::size_t>(size_);
        return DistributionInfo{
            policy_,
            num_partitions_,
            chunk,
            locality_id_
        };
    }

    // Access underlying data
    std::vector<T>& data() { return data_; }
    std::vector<T> const& data() const { return data_; }

    // Convert to local numpy array
    py::array_t<T> to_numpy() const {
        py::array_t<T> result(shape_);
        std::memcpy(result.mutable_data(), data_.data(),
                   data_.size() * sizeof(T));
        return result;
    }

    // Fill with value
    void fill(T value) {
        std::fill(data_.begin(), data_.end(), value);
    }

    // Element access
    T& operator[](std::size_t i) { return data_[i]; }
    T const& operator[](std::size_t i) const { return data_[i]; }

private:
    std::vector<py::ssize_t> shape_;
    DistributionPolicy policy_ = DistributionPolicy::None;
    py::ssize_t size_ = 0;
    std::size_t num_partitions_ = 1;
    std::uint32_t locality_id_ = 0;
    std::vector<T> data_;
};

// Type aliases for common types
using distributed_array_f64 = distributed_array<double>;
using distributed_array_f32 = distributed_array<float>;
using distributed_array_i64 = distributed_array<std::int64_t>;
using distributed_array_i32 = distributed_array<std::int32_t>;

}  // namespace hpxpy
