///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#include <hpx/compute_local/host/numa_domains.hpp>
#include <hpx/compute_local/host/target.hpp>
#include <hpx/resource_partitioner/detail/partitioner.hpp>
#include <hpx/runtime_local/get_os_thread_count.hpp>
#include <hpx/topology/cpu_mask.hpp>
#include <hpx/topology/topology.hpp>

#include <cstddef>
#include <vector>

namespace hpx::compute::host {

    std::vector<target> numa_domains()
    {
        auto const& topo = hpx::threads::create_topology();

        std::size_t numa_nodes = topo.get_number_of_numa_nodes();
        if (numa_nodes == 0)
        {
            numa_nodes = topo.get_number_of_sockets();
        }

        std::vector<hpx::threads::mask_type> node_masks(numa_nodes);
        for (auto& mask : node_masks)
        {
            hpx::threads::resize(mask, topo.get_number_of_pus());
        }

        auto const& rp = hpx::resource::get_partitioner();

        std::size_t const num_os_threads = hpx::get_os_thread_count();
        for (std::size_t num_thread = 0; num_thread != num_os_threads;
            ++num_thread)
        {
            std::size_t const pu_num = rp.get_pu_num(num_thread);
            std::size_t const numa_node = topo.get_numa_node_number(pu_num);

            auto const& mask = topo.get_thread_affinity_mask(pu_num);

            std::size_t const mask_size = hpx::threads::mask_size(mask);
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
                res.emplace_back(mask);
            }
        }

        return res;
    }
}    // namespace hpx::compute::host
