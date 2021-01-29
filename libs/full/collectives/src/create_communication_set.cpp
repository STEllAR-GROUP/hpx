//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/assert.hpp>
#include <hpx/collectives/communication_set.hpp>
#include <hpx/collectives/detail/communication_set_node.hpp>
#include <hpx/collectives/detail/communicator.hpp>
#include <hpx/modules/futures.hpp>
#include <hpx/naming_base/id_type.hpp>
#include <hpx/runtime_local/config_entry.hpp>

#include <climits>
#include <cstddef>
#include <cstdint>
#include <string>

namespace hpx { namespace lcos {

    // This function creates P/A communicator objects, (where P is the number
    // of participating sites, and A is the arity). Each invocation of the
    // function returns the communication node closest to the caller.
    /*
                     /                     \
                    0                       8
                   / \                     / \
                  /   \                   /   \
                 /     \                 /     \
                /       \               /       \
               /         \             /         \
              0           4           8           2
             / \         / \         / \         / \
            /   \       /   \       /   \       /   \
           0     2     4     6     8     0     2     4    <-- communicator nodes
          / \   / \   / \   / \   / \   / \   / \   / \
         0   1 2   3 4   5 6   7 8   9 0   1 2   3 4   5  <-- participants
    */
    // As can be seen from the graph, every evenly numbered participant should
    // create an communication node, while every odd numbered participant should
    // connect to its left neighbor (similar for other arities).
    //
    // The number of participants that will depend on a communication node is
    // calculated by counting the nodes that have a given node as its parent.
    //
    // The node some other node connects to is calculated by clearing the lowest
    // significant bit set (for arities == 2, similar for other arities).

    ///////////////////////////////////////////////////////////////////////////
    hpx::future<hpx::id_type> create_communication_set(char const* name,
        std::size_t num_sites, std::size_t this_site, std::size_t arity)
    {
        // set defaults for arguments
        if (num_sites == std::size_t(-1))
        {
            num_sites = static_cast<std::size_t>(
                hpx::get_num_localities(hpx::launch::sync));
        }
        if (this_site == std::size_t(-1))
        {
            this_site = static_cast<std::size_t>(hpx::get_locality_id());
        }
        if (arity == std::size_t(-1))
        {
            arity = std::stoull(
                get_config_entry("hpx.lcos.collectives.arity", "32"));
        }

        // the arity has to be a power of two but not equal to zero
        HPX_ASSERT(arity != 0 && detail::next_power_of_two(arity) == arity);

        hpx::future<hpx::id_type> node_id;
        if ((this_site % arity) == 0)
        {
            node_id = detail::create_communication_set_node(
                name, num_sites, this_site, arity);
        }
        else
        {
            node_id =
                hpx::find_from_basename(name, (this_site / arity) * arity);
        }

        return node_id;
    }
}}    // namespace hpx::lcos
#endif
