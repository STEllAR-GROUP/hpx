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

namespace hpx { namespace discovery { namespace stubs
{

struct discovery : components::stub_base<server::discovery>
{
    ///////////////////////////////////////////////////////////////////////////
    static lcos::future_value<std::vector<naming::id_type> >
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
    ) { return build_network(gid).get(); }

    ///////////////////////////////////////////////////////////////////////////
    static lcos::future_value<void> deploy_async(
        naming::id_type const& gid
      , std::map<naming::gid_type, std::size_t> const& m
    ) {
        typedef server::discovery::deploy_action action_type;
        return lcos::eager_future<action_type>(gid, m);
    }

    static void deploy_sync(
        naming::id_type const& gid
      , std::map<naming::gid_type, std::size_t> const& m
    ) { deploy_async(gid,m ).get(); }

    static void deploy(
        naming::id_type const& gid
      , std::map<naming::gid_type, std::size_t> const& m
    ) { deploy(gid, m).get(); }

    ///////////////////////////////////////////////////////////////////////////
    static lcos::future_value<hpx::uintptr_t> topology_lva_async(
        naming::id_type const& gid
    ) {
        typedef server::discovery::topology_lva_action action_type;
        return lcos::eager_future<action_type>(gid);
    }

    static hpx::uintptr_t topology_lva_sync(
        naming::id_type const& gid
    ) { return topology_lva_async(gid).get(); }

    static hpx::uintptr_t topology_lva(
        naming::id_type const& gid
    ) { return topology_lva(gid).get(); }

    ///////////////////////////////////////////////////////////////////////////
    static lcos::future_value<bool> empty_async(
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
    ) { return empty(gid).get(); }
};

}}}

#endif // HPX_ACFAA75C_4788_4006_8FC8_92682BBEB221

