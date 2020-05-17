//  Copyright (c) 2007-2020 Hartmut Kaiser
//  Copyright (c) 2020 Shahrzad Shirzad
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_SPLITTABLE_EXECUTOR_HPP
#define HPX_SPLITTABLE_EXECUTOR_HPP

#include <hpx/config.hpp>
#include <hpx/error.hpp>
#include <hpx/parallel/util/detail/splittable_task.hpp>
#include <hpx/traits/is_executor.hpp>

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
    struct splittable_executor : parallel_executor
    {
    public:
        /// Construct an \a splittable_executor executor parameters object
        ///
        /// \note Default constructed \a splittable_executor executor parameter
        ///       types will use 80 microseconds as the minimal time for which
        ///       any of the scheduled chunks should run.
        ///
        splittable_executor() {}

        splittable_executor(std::string exec_type)
        {
            if (exec_type != "all" && exec_type != "idle")
            {
                HPX_THROW_EXCEPTION(hpx::bad_parameter, "throw_hpx_exception",
                    "unknwn type, type should be either all or idle");
            }
            split_type_ = exec_type;
        }

        /// Construct an \a splittable_executor executor parameters object
        ///
        /// \param rel_time     [in] The time duration to use as the minimum
        ///                     to decide how many loop iterations should be
        ///                     combined.

        /// \cond NOINTERNAL
        // Estimate a chunk size based on number of cores used.
        template <typename Executor, typename F>
        std::size_t get_chunk_size(
            Executor&& exec, F&& f, std::size_t cores, std::size_t count)
        {
            return count;
        }
        /// \endcond

        template <typename F, typename S, typename... Ts>
        static std::vector<hpx::future<
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
                    std::forward<F>(f), elem, split_type_)));
            }

            return results;
        }

    private:
        friend class hpx::serialization::access;
        static std::string split_type_;
        /// \cond NOINTERNAL
    };

    std::string splittable_executor::split_type_ = "all";

    template <>
    struct is_bulk_two_way_executor<splittable_executor> : std::true_type
    {
    };

    template <typename Param, typename Exec, typename F>
    std::size_t get_chunk_size(
        Param& param, Exec& exec, F&& f, std::size_t core, std::size_t count)
    {
        return count;
    }

    /// \endcond
    template <typename AnyParameters, typename Executor>
    HPX_FORCEINLINE static std::size_t processing_units_count(
        AnyParameters&& params, Executor&& exec)
    {
        return hpx::get_os_thread_count();
    }
}}}    // namespace hpx::parallel::execution

#endif    //HPX_SPLITTABLE_EXECUTOR_HPP
