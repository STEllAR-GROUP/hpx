////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_1438B63A_BA4C_4DB3_B835_C8CDBD79B436)
#define HPX_1438B63A_BA4C_4DB3_B835_C8CDBD79B436

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/runtime/naming/name.hpp>

#include <examples/math/distributed/discovery/discovery.hpp>

namespace hpx { namespace balancing { namespace server
{

template <typename F, typename T>
struct HPX_COMPONENT_EXPORT integrator
    : components::managed_component_base<integrator<F> > 
{
    typedef components::managed_component_base<integrator> base_type; 
    
  private:
    topology_map const* topology_;
    boost::atomic<boost::uint32_t[2]> current_;

  public:
    enum actions
    {
        integrator_build_network,
        integrator_deploy,
        integrator_solve,
        integrator_regrid
    };

    std::vector<naming::id_type> build_network(
        std::vector<naming::id_type> const& discovery_network
      , T const& tolerance
      , T const& regrid_segs 
    ) {
        // IMPLEMENT
        return std::vector<naming::id_type>();
    }

    void deploy(
        T const& tolerance
      , T const& regrid_segs 
    ) {
        // IMPLEMENT
    }

    T solve(
        T const& lower_bound
      , T const& upper_bound
      , T const& segments
    ) {
        // IMPLEMENT
        return T(0);
    }

    T regrid(
        T const& lower_bound
      , T const& upper_bound
    ) {
        // IMPLEMENT
        return T(0);
    }

    typedef actions::result_action3<
        // class
        integrator
        // result
      , std::vector<naming::id_type>
        // action value type
      , integrator_build_network
        // arguments 
      , std::vector<naming::id_type> const&
      , T const& 
      , T const& 
        // function
      , &integrator::build_network
    > build_network_action;

    typedef actions::action1<
        // class
        integrator
        // action value type
      , integrator_deploy
        // arguments 
      , T const& 
      , T const& 
        // function
      , &integrator::deploy
    > deploy_action;
    
    typedef actions::result_action3<
        // class
        integrator
        // result
      , T 
        // action value type
      , integrator_regrid
        // arguments 
      , T const& 
      , T const& 
      , T const& 
        // function
      , &integrator::regrid
    > regrid_action;

    typedef actions::result_action2<
        // class
        integrator
        // result
      , T 
        // action value type
      , integrator_regrid
        // arguments 
      , T const& 
      , T const& 
        // function
      , &integrator::regrid
    > regrid_action;
};

}}}

#endif // HPX_1438B63A_BA4C_4DB3_B835_C8CDBD79B436

