//  Copyright (c) 2007-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/guided_chunk_size.hpp
/// \page hpx::execution::experimental::guided_chunk_size
/// \headerfile hpx/execution.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/execution/executors/execution_parameters.hpp>
#include <hpx/execution_base/traits/is_executor_parameters.hpp>
#include <hpx/serialization/serialize.hpp>
#include <hpx/timing/steady_clock.hpp>

#include <algorithm>
#include <cstddef>
#include <type_traits>

namespace hpx::execution::experimental {

    ///////////////////////////////////////////////////////////////////////////
    /// Iterations are dynamically assigned to threads in blocks as threads
    /// request those until no blocks remain to be assigned. Similar to
    /// \a dynamic_chunk_size except that the block size decreases each time a
    /// number of loop iterations is given to a thread. The size of the initial
    /// block is proportional to \a number_of_iterations / \a number_of_cores.
    /// Subsequent blocks are proportional to
    /// \a number_of_iterations_remaining / \a number_of_cores. The optional
    /// chunk size parameter defines the minimum block size. The default chunk
    /// size is 1.
    ///
    /// \note This executor parameters type is equivalent to OpenMP's GUIDED
    ///       scheduling directive.
    ///
    struct guided_chunk_size
    {
        /// Construct an \a dynamic_chunk_size executor parameters object
        ///
        /// \note Default constructed \a dynamic_chunk_size executor parameter
        ///       types will use a chunk size of '1'.
        ///
        guided_chunk_size() = default;

        /// Construct a \a guided_chunk_size executor parameters object
        ///
        /// \param min_chunk_size [in] The optional minimal chunk size to use
        ///                     as the minimal number of loop iterations to
        ///                     schedule together.
        ///                     The default minimal chunk size is 1.
        ///
        constexpr explicit guided_chunk_size(
            std::size_t min_chunk_size) noexcept
          : min_chunk_size_(min_chunk_size)
        {
        }

        /// \cond NOINTERNAL
        // This executor parameters type provides variable chunk sizes and
        // needs to be invoked for each of the chunks to be combined.
        using has_variable_chunk_size = std::true_type;

        template <typename Executor>
        friend constexpr std::size_t tag_override_invoke(
            hpx::parallel::execution::get_chunk_size_t,
            guided_chunk_size const& this_, Executor&& /* exec */,
            hpx::chrono::steady_duration const&, std::size_t cores,
            std::size_t num_tasks) noexcept
        {
            return (std::max)(
                this_.min_chunk_size_, (num_tasks + cores - 1) / cores);
        }
        /// \endcond

    private:
        /// \cond NOINTERNAL
        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive& ar, const unsigned int /* version */)
        {
            // clang-format off
            ar & min_chunk_size_;
            // clang-format on
        }
        /// \endcond

    private:
        /// \cond NOINTERNAL
        std::size_t min_chunk_size_ = 1;
        /// \endcond
    };
}    // namespace hpx::execution::experimental

/// \cond NOINTERNAL
template <>
struct hpx::parallel::execution::is_executor_parameters<
    hpx::execution::experimental::guided_chunk_size> : std::true_type
{
};
/// \endcond

namespace hpx::execution {

    using guided_chunk_size HPX_DEPRECATED_V(1, 9,
        "hpx::execution::guided_chunk_size is deprecated, use "
        "hpx::execution::experimental::guided_chunk_size instead") =
        hpx::execution::experimental::guided_chunk_size;
}
