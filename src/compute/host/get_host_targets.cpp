//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/async.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/runtime.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/get_os_thread_count.hpp>
#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/runtime/resource/detail/partitioner.hpp>
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/runtime/serialization/vector.hpp>
#include <hpx/runtime/threads/threadmanager.hpp>
#include <hpx/topology/topology.hpp>

#include <hpx/compute/host/target.hpp>

#include <cstddef>
#include <vector>

// this file requires serializing threads::mask_type
#if defined(HPX_HAVE_MORE_THAN_64_THREADS) || (defined(HPX_HAVE_MAX_CPU_COUNT) \
            && HPX_HAVE_MAX_CPU_COUNT > 64)
#  if defined(HPX_HAVE_MAX_CPU_COUNT)
#    include <bitset>
#    include <hpx/runtime/serialization/bitset.hpp>
#  else
#    include <boost/dynamic_bitset.hpp>
#    include <hpx/runtime/serialization/dynamic_bitset.hpp>
#  endif
#endif

namespace hpx { namespace compute { namespace host
{
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
}}}

HPX_PLAIN_ACTION(hpx::compute::host::get_local_targets,
    compute_host_get_targets_action);

namespace hpx { namespace compute { namespace host
{
    hpx::future<std::vector<target> > get_targets(hpx::id_type const& locality)
    {
        if (locality == hpx::find_here())
            return hpx::make_ready_future(get_local_targets());

        return hpx::async(compute_host_get_targets_action(), locality);
    }
}}}


