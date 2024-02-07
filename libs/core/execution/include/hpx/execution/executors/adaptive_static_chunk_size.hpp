//  Copyright (c) 2007-2023 Hartmut Kaiser
//  Copyright (c) 2022 Karame M.Shokooh
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/adaptive_static_chunk_size.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/execution/executors/execution_parameters.hpp>
#include <hpx/execution_base/traits/is_executor_parameters.hpp>
#include <hpx/serialization/serialize.hpp>
#include <hpx/timing/steady_clock.hpp>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <type_traits>

namespace hpx::execution::experimental {

    ///////////////////////////////////////////////////////////////////////////
    /// Loop iterations are divided into pieces of size \a chunk_size and then
    /// assigned to threads. If \a chunk_size is not specified, the iterations
    /// are evenly (if possible) divided contiguously among the threads.
    ///
    /// \note This executor parameters type is equivalent to OpenMP's STATIC
    ///       scheduling directive.
    ///
    struct adaptive_static_chunk_size
    {
        /// Construct a \a adaptive_static_chunk_size executor parameters object
        ///
        /// \note By default the number of loop iterations is determined from
        ///       the number of available cores and the overall number of loop
        ///       iterations to schedule.
        ///
        adaptive_static_chunk_size() = default;

        /// Construct a \a adaptive_static_chunk_size executor parameters object
        ///
        /// \param chunk_size   [in] The optional chunk size to use as the
        ///                     number of loop iterations to run on a single
        ///                     thread.
        ///
        constexpr explicit adaptive_static_chunk_size(
            std::size_t chunk_size) noexcept
          : chunk_size_(chunk_size)
        {
        }

        /// \cond NOINTERNAL
        template <typename Executor>
        friend std::size_t tag_override_invoke(
            hpx::parallel::execution::get_chunk_size_t,
            adaptive_static_chunk_size const& this_, Executor& exec,
            hpx::chrono::steady_duration const&, std::size_t cores,
            std::size_t input_size)
        {
            // Make sure the internal round-robin counter of the executor is
            // reset
            parallel::execution::reset_thread_distribution(this_, exec);

            // use the given chunk size if given
            if (this_.chunk_size_ != 0)
            {
                return this_.chunk_size_;
            }

            if (cores == 1)
            {
                return input_size;
            }

            // Return a chunk size that is a power of two; and that leads to at
            // least 2 chunks per core, and at most 4 chunks per core.
            std::size_t chunk_size = 1;
            std::size_t coeff = 1;

            if (input_size >= (1ull << 25))
            {
                coeff = 8;
            }
            else if (input_size >= (1ull << 19))
            {
                coeff = 4;    //-V112
            }
            while (chunk_size * coeff * cores < input_size)
            {
                chunk_size *= 2;
            }

            return chunk_size;
        }
        /// \endcond

    private:
        /// \cond NOINTERNAL
        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive& ar, const unsigned int /* version */)
        {
            // clang-format off
            ar & chunk_size_;
            // clang-format on
        }
        /// \endcond

    private:
        /// \cond NOINTERNAL
        std::size_t chunk_size_ = 0;
        /// \endcond
    };
}    // namespace hpx::execution::experimental

/// \cond NOINTERNAL
template <>
struct hpx::parallel::execution::is_executor_parameters<
    hpx::execution::experimental::adaptive_static_chunk_size> : std::true_type
{
};
/// \endcond

namespace hpx::execution {

    using adaptive_static_chunk_size HPX_DEPRECATED_V(1, 9,
        "hpx::execution::adaptive_static_chunk_size is deprecated, use "
        "hpx::execution::experimental::adaptive_static_chunk_size instead") =
        hpx::execution::experimental::adaptive_static_chunk_size;
}
