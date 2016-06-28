//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/async.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/runtime/threads/topology.hpp>
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/runtime/serialization/vector.hpp>

#include <hpx/compute/host/target.hpp>

#include <vector>

namespace hpx { namespace compute { namespace host
{
    std::vector<target> get_local_targets()
    {
        hpx::threads::topology const& topo = hpx::threads::get_topology();
        std::size_t numa_nodes = topo.get_number_of_numa_nodes();

        std::vector<target> targets;
        targets.reserve(numa_nodes);

        for(std::size_t i = 0; i != numa_nodes; ++i)
        {
            targets.emplace_back(
                target(topo.get_numa_node_affinity_mask_from_numa_node(i))
            );
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


