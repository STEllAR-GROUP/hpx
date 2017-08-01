///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#include <hpx/compute/host/numa_domains.hpp>
#include <hpx/compute/host/target.hpp>
#include <hpx/runtime.hpp>
#include <hpx/runtime/get_os_thread_count.hpp>
#include <hpx/runtime/resource_partitioner.hpp>
#include <hpx/runtime/threads/threadmanager.hpp>
#include <hpx/runtime/threads/topology.hpp>

#include <cstddef>
#include <vector>

namespace hpx { namespace compute { namespace host
{
    std::vector<target> numa_domains()
    {
        auto const& topo = hpx::threads::get_topology();

        std::size_t numa_nodes = topo.get_number_of_numa_nodes();
        if (numa_nodes == 0)
            numa_nodes = topo.get_number_of_sockets();
        std::vector<hpx::threads::mask_type> node_masks(numa_nodes);

        auto& rp = hpx::get_resource_partitioner();

        std::size_t num_os_threads = hpx::get_os_thread_count();
        for (std::size_t num_thread = 0; num_thread != num_os_threads;
             ++num_thread)
        {
            std::size_t pu_num = rp.get_pu_num(num_thread);
            std::size_t numa_node = topo.get_numa_node_number(pu_num);

            auto const& mask = topo.get_thread_affinity_mask(pu_num, true);

            std::size_t mask_size = hpx::threads::mask_size(mask);
            for (std::size_t idx = 0; idx != mask_size; ++idx)
            {
                if (hpx::threads::test(mask, idx))
                {
                    hpx::threads::set(node_masks[numa_node], idx);
                }
            }
        }

        // Sort out the masks which don't have any bits set
        std::vector<target> res;
        res.reserve(numa_nodes);

        for (auto& mask : node_masks)
        {
            if (hpx::threads::any(mask))
            {
                res.push_back(target(mask));
            }
        }

        return res;
    }
}}}
