////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_ACFAA75C_4788_4006_8FC8_92682BBEB221)
#define HPX_ACFAA75C_4788_4006_8FC8_92682BBEB221

#include <hpx/lcos/eager_future.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>

#include <examples/math/distributed/discovery/server/discovery.hpp>

namespace hpx { namespace balancing { namespace stubs
{

struct discovery : components::stub_base<server::discovery>
{
    ///////////////////////////////////////////////////////////////////////////
    static lcos::promise<std::vector<naming::id_type> >
    build_network_async(
        naming::id_type const& gid
    ) {
        typedef server::discovery::build_network_action action_type;
        return lcos::eager_future<action_type>(gid);
    }

    static std::vector<naming::id_type> build_network_sync(
        naming::id_type const& gid
    ) { return build_network_async(gid).get(); }

    static std::vector<naming::id_type> build_network(
        naming::id_type const& gid
    ) { return build_network_async(gid).get(); }

    ///////////////////////////////////////////////////////////////////////////
    static lcos::promise<std::size_t> topology_lva_async(
        naming::id_type const& gid
    ) {
        typedef server::discovery::topology_lva_action action_type;
        return lcos::eager_future<action_type>(gid);
    }

    static std::size_t topology_lva_sync(
        naming::id_type const& gid
    ) { return topology_lva_async(gid).get(); }

    static std::size_t topology_lva(
        naming::id_type const& gid
    ) { return topology_lva_async(gid).get(); }

    ///////////////////////////////////////////////////////////////////////////
    static lcos::promise<boost::uint32_t> total_shepherds_async(
        naming::id_type const& gid
    ) {
        typedef server::discovery::total_shepherds_action action_type;
        return lcos::eager_future<action_type>(gid);
    }

    static boost::uint32_t total_shepherds_sync(
        naming::id_type const& gid
    ) { return total_shepherds_async(gid).get(); }

    static boost::uint32_t total_shepherds(
        naming::id_type const& gid
    ) { return total_shepherds_async(gid).get(); }

    ///////////////////////////////////////////////////////////////////////////
    static lcos::promise<bool> empty_async(
        naming::id_type const& gid
    ) {
        typedef server::discovery::empty_action action_type;
        return lcos::eager_future<action_type>(gid);
    }

    static bool empty_sync(
        naming::id_type const& gid
    ) { return empty_async(gid).get(); }

    static bool empty(
        naming::id_type const& gid
    ) { return empty_async(gid).get(); }
};

}}}

#endif // HPX_ACFAA75C_4788_4006_8FC8_92682BBEB221

