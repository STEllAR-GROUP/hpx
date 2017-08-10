//  Copyright (c)      2017 Shoshana Jakobovits
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RESOURCE_PARTITIONER_FWD_AUG_10_2017_1005AM)
#define HPX_RESOURCE_PARTITIONER_FWD_AUG_10_2017_1005AM

#include <hpx/config.hpp>
#include <hpx/runtime/threads/policies/callback_notifier.hpp>
#include <hpx/runtime/threads_fwd.hpp>
#include <hpx/util/function.hpp>

#include <cstddef>
#include <memory>
#include <string>

namespace hpx
{
    namespace resource
    {
        class numa_domain;
        class core;
        class pu;

        class HPX_EXPORT partitioner;

        namespace detail
        {
            class HPX_EXPORT partitioner;
        }

        // May be used anywhere in code and returns a reference to the single,
        // global resource partitioner.
        HPX_EXPORT detail::partitioner& get_partitioner();

        // resource_partitioner mode
        enum partitioner_mode
        {
            mode_default = 0,
            mode_allow_oversubscription = 1,
            mode_allow_dynamic_pools = 2
        };

        using scheduler_function =
            util::function_nonser<
                std::unique_ptr<hpx::threads::detail::thread_pool_base>(
                    hpx::threads::policies::callback_notifier&,
                    std::size_t, std::size_t, std::size_t, std::string const&
                )>;

        // scheduler assigned to thread_pool
        // choose same names as in command-line options except with _ instead of -
        enum scheduling_policy
        {
            user_defined = -2,
            unspecified = -1,
            local = 0,
            local_priority_fifo = 1,
            local_priority_lifo = 2,
            static_ = 3,
            static_priority = 4,
            abp_priority = 5,
            hierarchy = 6,
            periodic_priority = 7,
            throttle = 8
        };
    }
}

#endif
