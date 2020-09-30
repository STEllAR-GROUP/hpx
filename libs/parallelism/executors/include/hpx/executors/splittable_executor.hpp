//  Copyright (c) 2007-2021 Hartmut Kaiser
//  Copyright (c) 2021 Shahrzad Shirzad
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/execution/traits/is_executor.hpp>
#include <hpx/executors/detail/splittable_task.hpp>
#include <hpx/modules/executors.hpp>
#include <hpx/modules/iterator_support.hpp>
#include <hpx/modules/serialization.hpp>
#include <hpx/modules/timing.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace parallel { namespace execution {
    ///////////////////////////////////////////////////////////////////////////
    /// Creates one splittable task containing all the loop iterations.
    /// The splittable task would be split into two parts if certain conditions
    /// are met, the smaller part would be executed by the current thread while
    /// the rest would be encapsulated into a splittable task. The size of the
    /// created task depends on the split type/mode, which could be guided or
    /// adaptive.

    struct splittable_executor
      : parallel_policy_executor<hpx::launch::async_policy>
    {
        using base_type = parallel_policy_executor<hpx::launch::async_policy>;

    public:
        /// Construct an \a splittable_executor executor parameters object
        /// current task. Split type is set to guided by default.

        splittable_executor()
          : split_type_(splittable_mode::guided)
          , min_task_size_(0)
        {
        }

        /// Construct an \a splittable_executor executor parameters object
        ///
        /// \param split_type_   [in] The split mode, to decide how to split the
        ///                     current task.

        splittable_executor(splittable_mode split_type)
          : split_type_(split_type)
          , min_task_size_(0)
        {
            if (split_type != splittable_mode::guided &&
                split_type != splittable_mode::adaptive)
            {
                HPX_THROW_EXCEPTION(hpx::bad_parameter,
                    "splittable_executor::splittable_executor",
                    "unknown type, type should be either guided, adaptive");
            }
        }

        /// Construct an \a splittable_executor executor parameters object
        ///
        /// \param split_type_    [in] The split mode, to decide how to split the
        ///                       current task.
        /// \param min_task_size_ [in] The split would only happen if size of
        ///                       the task that will be executed by the current
        ///                       thread after the split would be greater than
        ///                       min_task_size_.

        splittable_executor(
            splittable_mode split_type, std::size_t min_task_size)
          : split_type_(split_type)
          , min_task_size_(min_task_size)
        {
            if (split_type != splittable_mode::guided &&
                split_type != splittable_mode::adaptive)
            {
                HPX_THROW_EXCEPTION(hpx::bad_parameter,
                    "splittable_executor::splittable_executor",
                    "unknown type, type should be either guided, adaptive");
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
