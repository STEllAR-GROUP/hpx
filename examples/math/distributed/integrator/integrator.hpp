////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_11577A92_0DE4_4B05_87F1_E10DBB41A5ED)
#define HPX_11577A92_0DE4_4B05_87F1_E10DBB41A5ED

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/client_base.hpp>

#include <examples/math/distributed/integrator/stubs/integrator.hpp>

namespace hpx { namespace balancing 
{

template <typename T> 
struct integrator
    : components::client_base<integrator<T>, stubs::integrator<T> >
{
    typedef components::client_base<integrator<T>, stubs::integrator<T> >
        base_type;

  public:
    integrator(naming::id_type const& gid = naming::invalid_id) 
      : base_type(gid) {}

    ///////////////////////////////////////////////////////////////////////////
    lcos::future_value<std::vector<naming::id_type> > build_network_async(
        std::vector<naming::id_type> const& discovery_network
      , actions::function<T(T const&)> const& f
      , T const& tolerance
      , T const& regrid_segs 
    ) {
        return this->base_type::build_network_async
            (this->gid_, discovery_network, f, tolerance, regrid_segs);
    }

    std::vector<naming::id_type> build_network_sync(
        std::vector<naming::id_type> const& discovery_network
      , actions::function<T(T const&)> const& f
      , T const& tolerance
      , T const& regrid_segs 
    ) {
        return this->base_type::build_network_sync
            (this->gid_, discovery_network, f, tolerance, regrid_segs);
    }

    std::vector<naming::id_type> build_network(
        std::vector<naming::id_type> const& discovery_network
      , actions::function<T(T const&)> const& f
      , T const& tolerance
      , T const& regrid_segs 
    ) {
        return this->base_type::build_network
            (this->gid_, discovery_network, f, tolerance, regrid_segs);
    }

    ///////////////////////////////////////////////////////////////////////////
    lcos::future_value<void> deploy_async(
        naming::id_type const& discovery_gid
      , actions::function<T(T const&)> const& f
      , T const& tolerance
      , T const& regrid_segs 
    ) {
        return this->base_type::deploy_async
            (this->gid_, discovery_gid, f, tolerance, regrid_segs);
    }

    void deploy_sync(
        naming::id_type const& discovery_gid
      , actions::function<T(T const&)> const& f
      , T const& tolerance
      , T const& regrid_segs 
    ) {
        this->base_type::deploy_sync
            (this->gid_, discovery_gid, f, tolerance, regrid_segs);
    }

    void deploy(
        naming::id_type const& discovery_gid
      , actions::function<T(T const&)> const& f
      , T const& tolerance
      , T const& regrid_segs 
    ) {
        this->base_type::deploy
            (this->gid_, discovery_gid, f, tolerance, regrid_segs);
    }

    ///////////////////////////////////////////////////////////////////////////
    lcos::future_value<T> solve_async(
        T const& lower_bound
      , T const& upper_bound
      , T const& segments
    ) {
        return this->base_type::solve_async
            (this->gid_, lower_bound, upper_bound, segments);
    }

    T solve_sync(
        T const& lower_bound
      , T const& upper_bound
      , T const& segments
    ) {
        return this->base_type::solve_sync
            (this->gid_, lower_bound, upper_bound, segments);
    }

    T solve(
        T const& lower_bound
      , T const& upper_bound
      , T const& segments
    ) {
        return this->base_type::solve
            (this->gid_, lower_bound, upper_bound, segments);
    }

    ///////////////////////////////////////////////////////////////////////////
    lcos::future_value<T> regrid_async(
        T const& lower_bound
      , T const& upper_bound
    ) {
        return this->base_type::regrid_async
            (this->gid_, lower_bound, upper_bound);
    }

    T regrid_sync(
        T const& lower_bound
      , T const& upper_bound
    ) {
        return this->base_type::regrid_sync
            (this->gid_, lower_bound, upper_bound);
    }

    T regrid(
        T const& lower_bound
      , T const& upper_bound
    ) {
        return this->base_type::regrid(this->gid_, lower_bound, upper_bound);
    }
};

}}

#endif // HPX_11577A92_0DE4_4B05_87F1_E10DBB41A5ED

