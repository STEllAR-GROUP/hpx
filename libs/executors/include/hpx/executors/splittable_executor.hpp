//  Copyright (c) 2007-2020 Hartmut Kaiser
//  Copyright (c) 2020 Shahrzad Shirzad
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_SPLITTABLE_EXECUTOR_HPP
#define HPX_SPLITTABLE_EXECUTOR_HPP

#include <hpx/config.hpp>
#include <hpx/execution/traits/is_executor.hpp>
#include <hpx/executors/detail/splittable_task.hpp>
#include <hpx/include/async.hpp>
#include <hpx/modules/executors.hpp>
#include <hpx/modules/iterator_support.hpp>
#include <hpx/modules/serialization.hpp>
#include <hpx/modules/timing.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace hpx { namespace parallel { namespace execution {
    ///////////////////////////////////////////////////////////////////////////
    /// Loop iterations are divided into pieces and then assigned to threads.
    /// The number of loop iterations combined is determined based on
    /// measurements of how long the execution of 1% of the overall number of
    /// iterations takes.
    /// This executor parameters type makes sure that as many loop iterations
    /// are combined as necessary to run for the amount of time specified.
    ///
    struct splittable_executor
      : parallel_policy_executor<hpx::launch::async_policy>
    {
        using base_type = parallel_policy_executor<hpx::launch::async_policy>;

    public:
        /// Construct an \a splittable_executor executor parameters object
        ///
        /// \note Default constructed \a splittable_executor executor parameter
        ///       types will use 80 microseconds as the minimal time for which
        ///       any of the scheduled chunks should run.
        ///
        splittable_executor()
          : split_type_(splittable_mode::all)
          , min_task_size_(0)
        {
        }

        /// Construct an \a splittable_executor executor parameters object
        ///
        /// \param rel_time     [in] The time duration to use as the minimum
        ///                     to decide how many loop iterations should be
        ///                     combined.

        splittable_executor(splittable_mode split_type)
          : split_type_(split_type)
          , min_task_size_(0)
        {
            if (split_type != splittable_mode::all &&
                split_type != splittable_mode::idle &&
                split_type != splittable_mode::idle_mask &&
                split_type != splittable_mode::all_multiple_tasks)
            {
                HPX_THROW_EXCEPTION(hpx::bad_parameter,
                    "splittable_executor::splittable_executor",
                    "unknown type, type should be either all, idle, "
                    "idle_mask, or all_multiple_tasks");
            }
        }

        splittable_executor(
            splittable_mode split_type, std::size_t min_task_size)
          : split_type_(split_type)
          , min_task_size_(min_task_size)
        {
            if (split_type != splittable_mode::all &&
                split_type != splittable_mode::idle &&
                split_type != splittable_mode::idle_mask &&
                split_type != splittable_mode::all_multiple_tasks)
            {
                HPX_THROW_EXCEPTION(hpx::bad_parameter,
                    "splittable_executor::splittable_executor",
                    "unknown type, type should be either all, idle, "
                    "idle_mask, all_multiple_tasks");
            }
        }

        // Add two executor API functions that will be called before the
        // parallel algorithm starts executing and after it has finished
        // executing.
        //
        // Note that this method can cause problems if two parallel algorithms
        // are executed concurrently.
        template <typename Parameters>
        static void mark_begin_execution(Parameters&&)
        {
            hpx::threads::remove_scheduler_mode(
                hpx::threads::policies::enable_stealing);
        }

        template <typename Parameters>
        static void mark_end_execution(Parameters&&)
        {
            hpx::threads::add_scheduler_mode(
                hpx::threads::policies::enable_stealing);
        }
        
        /// \cond NOINTERNAL
        // Estimate a chunk size based on number of cores used.
        template <typename Parameters, typename F>
        static std::size_t get_chunk_size(
            Parameters&&, F&&, std::size_t, std::size_t count)
        {
            return count;
        }

        HPX_FORCEINLINE static std::size_t processing_units_count()
        {
            return hpx::get_os_thread_count();
        }
        /// \endcond

        template <typename F, typename S, typename... Ts>
        std::vector<hpx::future<
            typename detail::bulk_function_result<F, S, Ts...>::type>>
        bulk_async_execute(F&& f, S const& shape, Ts&&... ts)
        {
            std::vector<hpx::future<
                typename detail::bulk_function_result<F, S, Ts...>::type>>
                results;
            results.reserve(hpx::util::size(shape));

            for (auto const& elem : shape)
            {
                results.push_back(hpx::async(make_splittable_task(
                    static_cast<base_type&>(*this), std::forward<F>(f), elem,
                    split_type_, min_task_size_)));
            }

            return results;
        }

    private:
        friend class hpx::serialization::access;
        splittable_mode split_type_;
        std::size_t min_task_size_;
    };

    /// \cond NOINTERNAL
    template <>
    struct is_bulk_two_way_executor<splittable_executor> : std::true_type
    {
    };

#if HPX_VERSION_FULL < 0x010500
    // workaround for older HPX versions
    template <typename Param, typename Executor, typename F>
    std::size_t get_chunk_size(Param&& param, Executor&& exec, F&& f,
        std::size_t core, std::size_t count)
    {
        return count;
    }

    template <typename Param, typename Executor>
    HPX_FORCEINLINE static std::size_t processing_units_count(
        Param&& params, Executor&& exec)
    {
        return hpx::get_os_thread_count();
    }
#endif
    /// \endcond
}}}    // namespace hpx::parallel::execution

#endif    //HPX_SPLITTABLE_EXECUTOR_HPP