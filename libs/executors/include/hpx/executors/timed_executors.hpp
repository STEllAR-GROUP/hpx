//  Copyright (c) 2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_EXECUTORS_PARALLEL_TIMED_EXECUTORS_JAN_06_2017_0802AM)
#define HPX_EXECUTORS_PARALLEL_TIMED_EXECUTORS_JAN_06_2017_0802AM

#include <hpx/config.hpp>
#include <hpx/execution/executors/timed_executors.hpp>
#include <hpx/executors/parallel_executor.hpp>
#include <hpx/executors/sequenced_executor.hpp>
#include <hpx/execution/traits/executor_traits.hpp>

#include <type_traits>

namespace hpx { namespace parallel { namespace execution {
    /// \cond NOINTERNAL
    using sequenced_timed_executor =
        timed_executor<execution::sequenced_executor>;

    using parallel_timed_executor =
        timed_executor<execution::parallel_executor>;

    ///////////////////////////////////////////////////////////////////////////
    template <typename BaseExecutor>
    struct is_one_way_executor<
        parallel::execution::timed_executor<BaseExecutor>>
      : is_one_way_executor<typename std::decay<BaseExecutor>::type>
    {
    };

    template <typename BaseExecutor>
    struct is_two_way_executor<
        parallel::execution::timed_executor<BaseExecutor>>
      : is_two_way_executor<typename std::decay<BaseExecutor>::type>
    {
    };

    template <typename BaseExecutor>
    struct is_never_blocking_one_way_executor<
        parallel::execution::timed_executor<BaseExecutor>>
      : is_never_blocking_one_way_executor<
            typename std::decay<BaseExecutor>::type>
    {
    };

    ///////////////////////////////////////////////////////////////////////////
    template <>
    struct is_one_way_executor<parallel::execution::sequenced_timed_executor>
      : std::true_type
    {
    };

    template <>
    struct is_two_way_executor<parallel::execution::parallel_timed_executor>
      : std::true_type
    {
    };
    /// \endcond
}}}    // namespace hpx::parallel::execution
#endif
