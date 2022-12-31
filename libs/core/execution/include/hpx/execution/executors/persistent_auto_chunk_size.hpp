//  Copyright (c) 2016 Zahra Khatami
//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/persistent_auto_chunk_size.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/execution/executors/execution_parameters.hpp>
#include <hpx/execution_base/execution.hpp>
#include <hpx/execution_base/traits/is_executor_parameters.hpp>
#include <hpx/serialization/serialize.hpp>
#include <hpx/timing/high_resolution_clock.hpp>
#include <hpx/timing/steady_clock.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace hpx::execution::experimental {

    ///////////////////////////////////////////////////////////////////////////
    /// Loop iterations are divided into pieces and then assigned to threads.
    /// The number of loop iterations combined is determined based on
    /// measurements of how long the execution of 1% of the overall number of
    /// iterations takes.
    /// This executor parameters type makes sure that as many loop iterations
    /// are combined as necessary to run for the amount of time specified.
    ///
    struct persistent_auto_chunk_size
    {
    public:
        /// Construct an \a persistent_auto_chunk_size executor parameters object
        ///
        /// \note Default constructed \a persistent_auto_chunk_size executor parameter
        ///       types will use 0 microseconds as the execution time for each chunk
        ///       and 80 microseconds as the minimal time for which
        ///       any of the scheduled chunks should run.
        ///
        constexpr explicit persistent_auto_chunk_size(
            std::uint64_t num_iters_for_timing = 0) noexcept
          : chunk_size_time_(0)
          , min_time_(200000)
          , num_iters_for_timing_(num_iters_for_timing)
        {
        }

        /// Construct an \a persistent_auto_chunk_size executor parameters object
        ///
        /// \param time_cs      The execution time for each chunk.
        ///
        explicit persistent_auto_chunk_size(
            hpx::chrono::steady_duration const& time_cs,
            std::uint64_t num_iters_for_timing = 0) noexcept
          : chunk_size_time_(time_cs.value().count())
          , min_time_(200000)
          , num_iters_for_timing_(num_iters_for_timing)
        {
        }

        /// Construct an \a persistent_auto_chunk_size executor parameters object
        ///
        /// \param rel_time     [in] The time duration to use as the minimum
        ///                     to decide how many loop iterations should be
        ///                     combined.
        /// \param time_cs       The execution time for each chunk.
        ///
        persistent_auto_chunk_size(hpx::chrono::steady_duration const& time_cs,
            hpx::chrono::steady_duration const& rel_time,
            std::uint64_t num_iters_for_timing = 0) noexcept
          : chunk_size_time_(time_cs.value().count())
          , min_time_(rel_time.value().count())
          , num_iters_for_timing_(num_iters_for_timing)
        {
        }

        /// \cond NOINTERNAL
        // This executor parameters type synchronously invokes the provided
        // testing function in order to approximate the chunk-size.
        using invokes_testing_function = std::true_type;

        // Estimate execution time for one iteration
        template <typename Executor, typename F>
        auto measure_iteration(Executor&&, F&& f, std::size_t count)
        {
            // by default use 1% of the iterations
            if (num_iters_for_timing_ == 0)
            {
                num_iters_for_timing_ = count / 100;
            }

            // perform measurements only if necessary
            if (num_iters_for_timing_ > 0)
            {
                using hpx::chrono::high_resolution_clock;
                std::uint64_t t = high_resolution_clock::now();

                std::size_t test_chunk_size = f(num_iters_for_timing_);
                if (test_chunk_size != 0)
                {
                    if (chunk_size_time_ == 0)
                    {
                        t = (high_resolution_clock::now() - t) /
                            test_chunk_size;
                        chunk_size_time_ = t;
                    }
                    else
                    {
                        t = chunk_size_time_;
                    }

                    if (t != 0 && min_time_ >= t)
                    {
                        // return execution time for one iteration
                        return std::chrono::nanoseconds(t);
                    }
                }
            }

            return std::chrono::nanoseconds(0);
        }

        // Estimate a chunk size based on number of cores used.
        template <typename Executor>
        std::size_t get_chunk_size(Executor& /* exec */,
            hpx::chrono::steady_duration const& iteration_duration,
            std::size_t cores, std::size_t count) noexcept
        {
            // return chunk size which will create the required amount of work
            if (iteration_duration.value().count() != 0)
            {
                auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    iteration_duration.value());
                return (std::min)(count, (std::size_t)(min_time_ / ns.count()));
            }
            return (count + cores - 1) / cores;
        }
        /// \endcond

    private:
        /// \cond NOINTERNAL
        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive& ar, unsigned int const /* version */)
        {
            // clang-format off
            ar & chunk_size_time_ & min_time_ & num_iters_for_timing_;
            // clang-format on
        }
        /// \endcond

    private:
        /// \cond NOINTERNAL
        std::uint64_t chunk_size_time_;    // nanoseconds
        std::uint64_t min_time_;           // nanoseconds
        // number of iteration to use for timing
        std::uint64_t num_iters_for_timing_;
        /// \endcond
    };
}    // namespace hpx::execution::experimental

namespace hpx::parallel::execution {

    /// \cond NOINTERNAL
    template <>
    struct is_executor_parameters<
        hpx::execution::experimental::persistent_auto_chunk_size>
      : std::true_type
    {
    };
    /// \endcond
}    // namespace hpx::parallel::execution

namespace hpx::execution {

    using persistent_auto_chunk_size HPX_DEPRECATED_V(1, 9,
        "hpx::execution::persistent_auto_chunk_size is deprecated, use "
        "hpx::execution::experimental::persistent_auto_chunk_size instead") =
        hpx::execution::experimental::persistent_auto_chunk_size;
}
