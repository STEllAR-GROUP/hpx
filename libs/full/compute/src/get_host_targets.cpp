//  Copyright (c) 2016-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if defined(HPX_HAVE_DISTRIBUTED_RUNTIME)
#include <hpx/actions_base/plain_action.hpp>
#include <hpx/compute/host/distributed_target.hpp>
#include <hpx/compute/host/get_targets.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/modules/async_distributed.hpp>
#include <hpx/modules/serialization.hpp>
#include <hpx/naming_base/id_type.hpp>
#include <hpx/resource_partitioner/detail/partitioner.hpp>
#include <hpx/runtime_distributed/find_here.hpp>
#include <hpx/runtime_local/get_os_thread_count.hpp>
#include <hpx/runtime_local/runtime_local.hpp>
#include <hpx/topology/topology.hpp>

#include <vector>

HPX_PLAIN_ACTION(
    hpx::compute::host::get_local_targets, compute_host_get_targets_action)

namespace hpx::compute::host::distributed {

    namespace detail {

        std::vector<host::distributed::target> get_remote_targets(
            std::vector<host::target> const& targets)
        {
            std::vector<host::distributed::target> remote_targets;
            remote_targets.reserve(targets.size());
            for (auto const& t : targets)
            {
                remote_targets.emplace_back(t);
            }
            return remote_targets;
        }
    }    // namespace detail

    hpx::future<std::vector<host::distributed::target>> get_targets(
        hpx::id_type const& locality)
    {
        if (locality == hpx::find_here())
        {
            return hpx::make_ready_future(
                detail::get_remote_targets(get_local_targets()));
        }

        return hpx::async(compute_host_get_targets_action(), locality)
            .then([](auto&& f) { return detail::get_remote_targets(f.get()); });
    }
}    // namespace hpx::compute::host::distributed

#endif
