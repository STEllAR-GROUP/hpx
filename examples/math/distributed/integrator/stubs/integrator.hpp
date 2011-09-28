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

template <typename T> 
struct integrator : components::stub_base<server::integrator<T> >
{
    ///////////////////////////////////////////////////////////////////////////
    static lcos::promise<std::vector<naming::id_type> >
    build_network_async(
        naming::id_type const& gid
      , std::vector<naming::id_type> const& discovery_network
      , actions::function<T(T const&)> const& f
      , T const& tolerance
      , boost::uint32_t regrid_segs 
      , T const& eps 
    ) {
        typedef typename server::integrator<T>::build_network_action
            action_type;
        return lcos::eager_future<action_type>
            (gid, discovery_network, f, tolerance, regrid_segs, eps);
    }

    static std::vector<naming::id_type> build_network_sync(
        naming::id_type const& gid
      , std::vector<naming::id_type> const& discovery_network
      , actions::function<T(T const&)> const& f
      , T const& tolerance
      , boost::uint32_t regrid_segs 
      , T const& eps 
    ) {
        return build_network_async
            (gid, discovery_network, f, tolerance, regrid_segs, eps).get();
    }

    static std::vector<naming::id_type> build_network(
        naming::id_type const& gid
      , std::vector<naming::id_type> const& discovery_network
      , actions::function<T(T const&)> const& f
      , T const& tolerance
      , boost::uint32_t regrid_segs 
      , T const& eps 
    ) {
        return build_network_async
            (gid, discovery_network, f, tolerance, regrid_segs, eps).get();
    }

    ///////////////////////////////////////////////////////////////////////////
    static lcos::promise<T>
    solve_async(
        naming::id_type const& gid
      , T const& lower_bound
      , T const& upper_bound
      , boost::uint32_t segments
    ) {
        typedef typename server::integrator<T>::solve_action
            action_type;
        return lcos::eager_future<action_type>
            (gid, lower_bound, upper_bound, segments, 0);
    }

    static T solve_sync(
        naming::id_type const& gid
      , T const& lower_bound
      , T const& upper_bound
      , boost::uint32_t segments
    ) {
        return solve_async(gid, lower_bound, upper_bound, segments).get();
    }

    static T solve(
        naming::id_type const& gid
      , T const& lower_bound
      , T const& upper_bound
      , boost::uint32_t segments
    ) {
        return solve_async(gid, lower_bound, upper_bound, segments).get();
    }
};

}}}

#endif // HPX_A1D649B9_226D_4DE3_B5D9_C3C537186DF9

