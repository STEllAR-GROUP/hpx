////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach and Katelyn Kufahl
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_1438B63A_BA4C_4DB3_B835_C8CDBD79B436)
#define HPX_1438B63A_BA4C_4DB3_B835_C8CDBD79B436

#include <list>

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/eager_future.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/components/stubs/runtime_support.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/runtime/actions/function.hpp>
#include <hpx/runtime/naming/name.hpp>

#include <examples/math/distributed/discovery/discovery.hpp>

namespace hpx { namespace balancing { namespace server
{

template <typename T> 
struct HPX_COMPONENT_EXPORT integrator
    : components::managed_component_base<integrator<T> > 
{
    typedef components::managed_component_base<integrator> base_type; 
   
    struct current_shepherd
    {
        boost::uint32_t prefix;
        boost::uint32_t shepherd;
    };
     
  private:
    topology_map const* topology_;
    boost::atomic<current_shepherd> current_;
    actions::function<T(T const&)> f_;
    T tolerance_;
    T regrid_segs_;

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
      , actions::function<T(T const&)> const& f 
      , T const& tolerance
      , T const& regrid_segs 
    ) {
        std::list<lcos::future_value<naming::id_type, naming::gid_type> >
            results0; 
    
        BOOST_FOREACH(naming::gid_type const& node, discovery_network)
        {
            results0.push_back
                (components::stubs::runtime_support::create_component_async
                    (locality, components::get_component_type<integrator>()));
        }
    
        std::vector<naming::id_type> integrator_network;

        network.push_back(this->get_gid(this));
    
        typedef lcos::future_value<naming::id_type, naming::gid_type>
            gid_future;
        BOOST_FOREACH(gid_future const& r, results0)
        { integrator_network.push_back(r.get()); }

        BOOST_ASSERT(integrator_network.size() == discovery_network.size());
    
        std::list<lcos::future_value<void> > results1;
   
        for (std::size_t i = 0; i < integrator_network.size(); ++i)
        {
            typedef lcos::future_value<deploy_action> deploy_future; 
            results1.push_back(deploy_future(integrator_network[i]
                                           , discovery_network[i]
                                           , f
                                           , tolerance
                                           , regrid_segs));
        }
    
        BOOST_FOREACH(lcos::future_value<void> const& r, results1)
        { r.get(); }
    
        return integrator_network;
    }

    void deploy(
        naming::id_type const& discovery
      , actions::function<T(T const&)> const& f 
      , T const& tolerance
      , T const& regrid_segs 
    ) {
        BOOST_ASSERT(applier::get_prefix_id() ==
                     naming::get_prefix_from_gid(discovery));

        discovery disc_client(discovery);

        // DMA shortcut to reduce scheduling overhead.
        topology_ = reinterpret_cast<topology_map const*>
            (disc_client.topology_lva_sync());

        f_ = f;
        tolerance_ = tolerance;
        regrid_segs_ = regrid_segs; 

        // IMPLEMENT: set current
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

    typedef actions::result_action4<
        // class
        integrator<T>
        // result
      , std::vector<naming::id_type>
        // action value type
      , integrator_build_network
        // arguments 
      , std::vector<naming::id_type> const&
      , actions::function<T(T const&)> const&
      , T const& 
      , T const& 
        // function
      , &integrator<T>::build_network
    > build_network_action;

    typedef actions::action4<
        // class
        integrator<T>
        // action value type
      , integrator_deploy
        // arguments 
      , naming::id_type const&
      , actions::function<T(T const&)> const&
      , T const& 
      , T const& 
        // function
      , &integrator<T>::deploy
    > deploy_action;
    
    typedef actions::result_action3<
        // class
        integrator<T>
        // result
      , T 
        // action value type
      , integrator_solve
        // arguments 
      , T const& 
      , T const& 
      , T const& 
        // function
      , &integrator<T>::solve
    > solve_action;

    typedef actions::result_action2<
        // class
        integrator<T>
        // result
      , T 
        // action value type
      , integrator_regrid
        // arguments 
      , T const& 
      , T const& 
        // function
      , &integrator<T>::regrid
    > regrid_action;
};

}}}

#endif // HPX_1438B63A_BA4C_4DB3_B835_C8CDBD79B436

