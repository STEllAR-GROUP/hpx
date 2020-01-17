//  Copyright (c) 2007-2018 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/pool_executor.hpp

#if !defined(HPX_PARALLEL_EXECUTORS_POOL_EXECUTOR_FEB_17_2018_0327PM)
#define HPX_PARALLEL_EXECUTORS_POOL_EXECUTOR_FEB_17_2018_0327PM

#include <hpx/config.hpp>
#include <hpx/execution/executors/thread_pool_executor.hpp>

// TODO: Move this file elsewhere or remove these dependencies
#include <hpx/runtime_fwd.hpp>
#include <hpx/threadmanager.hpp>

#include <string>
#include <type_traits>

namespace hpx { namespace parallel { namespace execution {
    ///////////////////////////////////////////////////////////////////////////
    class HPX_EXPORT pool_executor : public thread_pool_executor
    {
    public:
        pool_executor(std::string const& pool_name,
            threads::thread_stacksize stacksize =
                threads::thread_stacksize_default)
          : thread_pool_executor(
                &threads::get_thread_manager().get_pool(pool_name),
                threads::thread_priority_default, stacksize)
        {
        }

        pool_executor(std::string const& pool_name,
            threads::thread_priority priority,
            threads::thread_stacksize stacksize =
                threads::thread_stacksize_default)
          : thread_pool_executor(
                &hpx::threads::get_thread_manager().get_pool(pool_name),
                priority, stacksize)
        {
        }
    };

    /// \cond NOINTERNAL
    template <>
    struct is_one_way_executor<pool_executor> : std::true_type
    {
    };

    template <>
    struct is_two_way_executor<pool_executor> : std::true_type
    {
    };

    template <>
    struct is_bulk_two_way_executor<pool_executor> : std::true_type
    {
    };
    /// \endcond
}}}    // namespace hpx::parallel::execution

#endif
