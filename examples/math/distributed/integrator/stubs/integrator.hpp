////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_A1D649B9_226D_4DE3_B5D9_C3C537186DF9)
#define HPX_A1D649B9_226D_4DE3_B5D9_C3C537186DF9

#include <hpx/lcos/eager_future.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>

#include <examples/math/distributed/integrator/server/integrator.hpp>

namespace hpx { namespace balancing { namespace stubs
{

template <typename F>
struct integrator : components::stub_base<server::integrator<F, T> >
{
    ///////////////////////////////////////////////////////////////////////////
    static lcos::future_value<std::vector<naming::id_type> >
    build_network_async(
        naming::id_type const& gid
      , std::vector<naming::id_type> const& discovery_network
      , T const& tolerance
      , T const& regrid_segs 
    ) {
        typedef typename server::integrator<F, T>::build_network_action
            action_type;
        return lcos::eager_future<action_type>
            (gid, discovery_network, tolerance, regrid_segs);
    }

    static std::vector<naming::id_type> build_network_sync(
        naming::id_type const& gid
      , std::vector<naming::id_type> const& discovery_network
      , T const& tolerance
      , T const& regrid_segs 
    ) {
        return build_network_async
            (gid, discovery_network, tolerance, regrid_segs).get();
    }

    static std::vector<naming::id_type> build_network(
        naming::id_type const& gid
      , std::vector<naming::id_type> const& discovery_network
      , T const& tolerance
      , T const& regrid_segs 
    ) {
        return build_network_async
            (gid, discovery_network, tolerance, regrid_segs).get();
    }

    ///////////////////////////////////////////////////////////////////////////
    static lcos::future_value<void> deploy_async(
        naming::id_type const& gid
      , T const& tolerance
      , T const& regrid_segs 
    ) {
        typedef typename server::integrator<F, T>::deploy_action action_type;
        return lcos::eager_future<action_type>(gid, tolerance, regrid_segs);
    }

    static void deploy_sync(
        naming::id_type const& gid
      , T const& tolerance
      , T const& regrid_segs 
    ) {
        deploy_async(gid, tolerance, regrid_segs).get();
    }

    static void deploy(
        naming::id_type const& gid
      , T const& tolerance
      , T const& regrid_segs 
    ) {
        deploy_async(gid, tolerance, regrid_segs).get();
    }

    ///////////////////////////////////////////////////////////////////////////
    static lcos::future_value<T>
    solve_async(
        naming::id_type const& gid
      , T const& lower_bound
      , T const& upper_bound
      , T const& segments
    ) {
        typedef typename server::integrator<F, T>::solve_action
            action_type;
        return lcos::eager_future<action_type>
            (gid, lower_bound, upper_bound, segments);
    }

    static T solve_sync(
        naming::id_type const& gid
      , T const& lower_bound
      , T const& upper_bound
      , T const& segments
    ) {
        return solve_async
            (gid, lower_bound, upper_bound, segments).get();
    }

    static T solve(
        naming::id_type const& gid
      , T const& lower_bound
      , T const& upper_bound
      , T const& segments
    ) {
        return solve_async
            (gid, lower_bound, upper_bound, segments).get();
    }

    ///////////////////////////////////////////////////////////////////////////
    static lcos::future_value<T>
    regrid_async(
        naming::id_type const& gid
      , T const& lower_bound
      , T const& upper_bound
      , T const& segments
    ) {
        typedef typename server::integrator<F, T>::regrid_action
            action_type;
        return lcos::eager_future<action_type>(gid, lower_bound, upper_bound);
    }

    static T regrid_sync(
        naming::id_type const& gid
      , T const& lower_bound
      , T const& upper_bound
      , T const& segments
    ) {
        return regrid_async(gid, lower_bound, upper_bound).get();
    }

    static T regrid(
        naming::id_type const& gid
      , T const& lower_bound
      , T const& upper_bound
    ) {
        return regrid_async(gid, lower_bound, upper_bound).get();
    }
};

}}}

#endif // HPX_A1D649B9_226D_4DE3_B5D9_C3C537186DF9

