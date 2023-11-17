//  Copyright (c) 2007-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/default_parameters.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/execution/executors/execution_parameters_fwd.hpp>
#include <hpx/execution_base/traits/is_executor_parameters.hpp>
#include <hpx/timing/steady_clock.hpp>

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
    struct default_parameters
    {
        /// Construct a \a default_parameters executor parameters object
        ///
        /// \note By default the number of loop iterations is determined from
        ///       the number of available cores and the overall number of loop
        ///       iterations to schedule.
        ///
        default_parameters() = default;

        /// \cond NOINTERNAL
        template <typename Executor>
        std::size_t get_chunk_size(Executor&& exec,
            hpx::chrono::steady_duration const&, std::size_t cores,
            std::size_t num_tasks)
        {
            // Make sure the internal round-robin counter of the executor is
            // reset
            parallel::execution::reset_thread_distribution(
                *this, HPX_FORWARD(Executor, exec));

            if (cores == 1)
            {
                return num_tasks;
            }

            // Return a chunk size that is a power of two; and that leads to at
            // least 2 chunks per core, and at most 4 chunks per core.
            std::size_t chunk_size = 1;
            while (chunk_size * cores * 4 < num_tasks)    //-V112
            {
                chunk_size *= 2;
            }

            return chunk_size;
        }
        /// \endcond
    };
}    // namespace hpx::execution::experimental

/// \cond NOINTERNAL
template <>
struct hpx::parallel::execution::is_executor_parameters<
    hpx::execution::experimental::default_parameters> : std::true_type
{
};
/// \endcond
