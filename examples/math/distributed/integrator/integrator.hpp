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
    lcos::promise<std::vector<naming::id_type> > build_network_async(
        std::vector<naming::id_type> const& discovery_network
      , hpx::actions::function<T(T const&)> const& f
      , T const& tolerance
      , boost::uint32_t regrid_segs 
      , T const& eps 
    ) {
        return this->base_type::build_network_async
            (this->gid_, discovery_network, f, tolerance, regrid_segs, eps);
    }

    std::vector<naming::id_type> build_network_sync(
        std::vector<naming::id_type> const& discovery_network
      , hpx::actions::function<T(T const&)> const& f
      , T const& tolerance
      , boost::uint32_t regrid_segs 
      , T const& eps 
    ) {
        return this->base_type::build_network_sync
            (this->gid_, discovery_network, f, tolerance, regrid_segs, eps);
    }

    std::vector<naming::id_type> build_network(
        std::vector<naming::id_type> const& discovery_network
      , hpx::actions::function<T(T const&)> const& f
      , T const& tolerance
      , boost::uint32_t regrid_segs 
      , T const& eps 
    ) {
        return this->base_type::build_network
            (this->gid_, discovery_network, f, tolerance, regrid_segs, eps);
    }

    ///////////////////////////////////////////////////////////////////////////
    lcos::promise<T> solve_async(
        T const& lower_bound
      , T const& upper_bound
      , boost::uint32_t segments
    ) {
        return this->base_type::solve_async
            (this->gid_, lower_bound, upper_bound, segments);
    }

    T solve_sync(
        T const& lower_bound
      , T const& upper_bound
      , boost::uint32_t segments
    ) {
        return this->base_type::solve_sync
            (this->gid_, lower_bound, upper_bound, segments);
    }

    T solve(
        T const& lower_bound
      , T const& upper_bound
      , boost::uint32_t segments
    ) {
        return this->base_type::solve
            (this->gid_, lower_bound, upper_bound, segments);
    }
};

}}

#endif // HPX_11577A92_0DE4_4B05_87F1_E10DBB41A5ED

