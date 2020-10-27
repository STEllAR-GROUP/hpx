//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/compute/host/target.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/resource_partitioner/detail/partitioner.hpp>
#include <hpx/runtime_local/get_os_thread_count.hpp>
#include <hpx/runtime_local/runtime_local.hpp>
#include <hpx/serialization/serialize.hpp>
#include <hpx/serialization/vector.hpp>
#include <hpx/topology/topology.hpp>

#if defined(HPX_HAVE_DISTRIBUTED_RUNTIME)
#include <hpx/actions_base/plain_action.hpp>
#include <hpx/modules/async_distributed.hpp>
#include <hpx/naming_base/id_type.hpp>
#endif

#include <cstddef>
#include <vector>

// this file requires serializing threads::mask_type
#if defined(HPX_HAVE_MORE_THAN_64_THREADS) ||                                  \
    (defined(HPX_HAVE_MAX_CPU_COUNT) && HPX_HAVE_MAX_CPU_COUNT > 64)
#if defined(HPX_HAVE_MAX_CPU_COUNT)
#include <hpx/serialization/bitset.hpp>
#include <bitset>
#else
#include <hpx/serialization/dynamic_bitset.hpp>
#include <boost/dynamic_bitset.hpp>
#endif
#endif

namespace hpx { namespace compute { namespace host {
    std::vector<target> get_local_targets()
    {
        std::size_t num_os_threads = hpx::get_os_thread_count();

        std::vector<target> targets;
        targets.reserve(num_os_threads);

        auto& rp = hpx::resource::get_partitioner();
        for (std::size_t num_thread = 0; num_thread != num_os_threads;
             ++num_thread)
        {
            targets.emplace_back(rp.get_pu_mask(num_thread));
        }

        return targets;
    }
}}}    // namespace hpx::compute::host

#if defined(HPX_HAVE_DISTRIBUTED_RUNTIME)
HPX_PLAIN_ACTION(
    hpx::compute::host::get_local_targets, compute_host_get_targets_action);

namespace hpx { namespace compute { namespace host {
    hpx::future<std::vector<target>> get_targets(hpx::id_type const& locality)
    {
        if (locality == hpx::find_here())
            return hpx::make_ready_future(get_local_targets());

        return hpx::async(compute_host_get_targets_action(), locality);
    }
}}}    // namespace hpx::compute::host
#endif
