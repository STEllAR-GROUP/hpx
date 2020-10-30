//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/static_chunk_size.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/execution/traits/is_executor_parameters.hpp>
#include <hpx/serialization/serialize.hpp>

#include <hpx/execution/executors/execution_parameters_fwd.hpp>

#include <cstddef>
#include <type_traits>

namespace hpx { namespace execution {
    ///////////////////////////////////////////////////////////////////////////
    /// Loop iterations are divided into pieces of size \a chunk_size and then
    /// assigned to threads. If \a chunk_size is not specified, the iterations
    /// are evenly (if possible) divided contiguously among the threads.
    ///
    /// \note This executor parameters type is equivalent to OpenMP's STATIC
    ///       scheduling directive.
    ///
    struct static_chunk_size
    {
        /// Construct a \a static_chunk_size executor parameters object
        ///
        /// \note By default the number of loop iterations is determined from
        ///       the number of available cores and the overall number of loop
        ///       iterations to schedule.
        ///
        constexpr static_chunk_size()
          : chunk_size_(0)
        {
        }

        /// Construct a \a static_chunk_size executor parameters object
        ///
        /// \param chunk_size   [in] The optional chunk size to use as the
        ///                     number of loop iterations to run on a single
        ///                     thread.
        ///
        constexpr explicit static_chunk_size(std::size_t chunk_size)
          : chunk_size_(chunk_size)
        {
        }

        /// \cond NOINTERNAL
        template <typename Executor, typename F>
        std::size_t get_chunk_size(
            Executor& exec, F&&, std::size_t cores, std::size_t num_tasks)
        {
            // Make sure the internal round robin counter of the executor is
            // reset
            parallel::execution::reset_thread_distribution(*this, exec);

            // use the given chunk size if given
            if (chunk_size_ != 0)
            {
                return chunk_size_;
            }

            if (cores == 1)
            {
                return num_tasks;
            }

            // Return a chunk size that is a power of two; and that leads to at
            // least 2 chunks per core, and at most 4 chunks per core.
            std::size_t chunk_size = 1;
            while (chunk_size * cores * 4 < num_tasks)
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
            ar& chunk_size_;
        }
        /// \endcond

    private:
        /// \cond NOINTERNAL
        std::size_t chunk_size_;
        /// \endcond
    };
}}    // namespace hpx::execution

namespace hpx { namespace parallel { namespace execution {
    using static_chunk_size HPX_DEPRECATED_V(1, 6,
        "hpx::parallel::execution::static_chunk_size is deprecated. Use "
        "hpx::execution::static_chunk_size instead.") =
        hpx::execution::static_chunk_size;
}}}    // namespace hpx::parallel::execution

namespace hpx { namespace parallel { namespace execution {
    /// \cond NOINTERNAL
    template <>
    struct is_executor_parameters<hpx::execution::static_chunk_size>
      : std::true_type
    {
    };
    /// \endcond
}}}    // namespace hpx::parallel::execution
