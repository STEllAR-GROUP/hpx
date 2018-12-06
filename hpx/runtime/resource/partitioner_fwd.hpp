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

        class partitioner;

        namespace detail
        {
            class HPX_EXPORT partitioner;
            void HPX_EXPORT delete_partitioner();
        }

        /// May be used anywhere in code and returns a reference to the single,
        /// global resource partitioner.
        HPX_EXPORT detail::partitioner& get_partitioner();

        /// Returns true if the resource partitioner has been initialized.
        /// Returns false otherwise.
        HPX_EXPORT bool is_partitioner_valid();

        /// This enumeration describes the modes available when creating a
        /// resource partitioner.
        enum partitioner_mode
        {
            /// Default mode.
            mode_default = 0,
            /// Allow processing units to be oversubscribed, i.e. multiple
            /// worker threads to share a single processing unit.
            mode_allow_oversubscription = 1,
            /// Allow worker threads to be added and removed from thread pools.
            mode_allow_dynamic_pools = 2
        };

        using scheduler_function =
            util::function_nonser<
                std::unique_ptr<hpx::threads::thread_pool_base>(
                    hpx::threads::policies::callback_notifier&,
                    std::size_t, std::size_t, std::size_t, std::string const&
                )>;

        // Choose same names as in command-line options except with _ instead of
        // -.

        /// This enumeration lists the available scheduling policies (or
        /// schedulers) when creating thread pools.
        enum scheduling_policy
        {
            user_defined = -2,
            unspecified = -1,
            local = 0,
            local_priority_fifo = 1,
            local_priority_lifo = 2,
            static_ = 3,
            static_priority = 4,
            abp_priority_fifo = 5,
            abp_priority_lifo = 6,
            shared_priority = 7,
        };
    }
}

#endif
