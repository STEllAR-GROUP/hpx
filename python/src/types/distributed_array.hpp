// HPXPy - Distribution utilities
//
// SPDX-License-Identifier: BSL-1.0
//
// This header provides distribution policy definitions and utilities
// for Phase 3 distributed array support.

#pragma once

#include <pybind11/pybind11.h>

#include <cstdint>
#include <string>

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
    // In single-locality mode, this is always 0
    return 0;
}

// Get number of localities
inline std::size_t get_num_localities() {
    // In single-locality mode, this is always 1
    return 1;
}

}  // namespace hpxpy
