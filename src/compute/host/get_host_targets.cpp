//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/async.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/runtime/get_os_thread_count.hpp>
#include <hpx/runtime.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/runtime/threads/topology.hpp>
#include <hpx/runtime/threads/threadmanager.hpp>
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/runtime/serialization/vector.hpp>

#include <hpx/compute/host/target.hpp>

#include <vector>

namespace hpx { namespace compute { namespace host
{
    std::vector<target> get_local_targets()
    {
        hpx::threads::topology const& topo = hpx::threads::get_topology();
        std::size_t num_os_threads = hpx::get_os_thread_count();

        std::vector<target> targets;
        targets.reserve(num_os_threads);

        auto & tm = hpx::get_runtime().get_thread_manager();
        for(std::size_t num_thread = 0; num_thread != num_os_threads; ++num_thread)
        {
            std::size_t pu_num = tm.get_pu_num(num_thread);

            auto const& mask = topo.get_thread_affinity_mask(pu_num, true);
            targets.emplace_back(mask);
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


