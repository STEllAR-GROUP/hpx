//  Copyright (c) 2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/collect_chunking_parameters.hpp
/// \page hpx::execution::experimental::collect_chunking_parameters
/// \headerfile hpx/execution.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/execution/executors/execution_parameters.hpp>
#include <hpx/execution_base/traits/is_executor_parameters.hpp>
#include <hpx/serialization/serialize.hpp>

#include <cstddef>
#include <type_traits>

namespace hpx::execution::experimental {

    /// Collected execution parameters
    struct chunking_parameters
    {
        std::size_t num_elements;
        std::size_t num_cores;
        std::size_t num_chunks;
        std::size_t chunk_size;

        template <typename Archive>
        void serialize(Archive& ar, unsigned int const)
        {
            ar & num_elements & num_cores & num_chunks & chunk_size;
        }
    };

    /// Collect various parameters used for running a parallel algorithm
    struct collect_chunking_parameters
    {
        explicit constexpr collect_chunking_parameters(
            chunking_parameters& exec_params) noexcept
          : exec_params_(&exec_params)
        {
        }

        /// \cond NOINTERNAL
        template <typename Executor>
        friend void tag_override_invoke(
            hpx::execution::experimental::collect_execution_parameters_t,
            collect_chunking_parameters const& this_, Executor&&,
            std::size_t const num_elements, std::size_t const num_cores,
            std::size_t const num_chunks, std::size_t const chunk_size) noexcept
        {
            this_.exec_params_->num_elements = num_elements;
            this_.exec_params_->num_cores = num_cores;
            this_.exec_params_->num_chunks = num_chunks;
            this_.exec_params_->chunk_size = chunk_size;
        }
        /// \endcond

    private:
        /// \cond NOINTERNAL
        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive& ar, unsigned int const)
        {
            // clang-format off
            ar & *exec_params_;
            // clang-format on
        }

        chunking_parameters* exec_params_;
        /// \endcond
    };

    /// \cond NOINTERNAL
    template <>
    struct is_executor_parameters<collect_chunking_parameters> : std::true_type
    {
    };
    /// \endcond
}    // namespace hpx::execution::experimental
