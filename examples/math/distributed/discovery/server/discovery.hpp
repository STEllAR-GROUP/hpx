////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_17581FEA_9F85_45E9_8CB1_36AF08A4809B)
#define HPX_17581FEA_9F85_45E9_8CB1_36AF08A4809B

#include <vector>
#include <map>

#include <hpx/config.hpp>
#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/local_shared_mutex.hpp>
#include <hpx/runtime/components/server/simple_component_base.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/util/uintptr_t.hpp>

namespace hpx { namespace balancing { namespace server
{

struct HPX_COMPONENT_EXPORT discovery
    : components::simple_component_base<discovery> 
{
    typedef components::simple_component_base<discovery> base_type; 
    // }}}
    
    static std::size_t report_shepherd_count();

  private:
    std::map<naming::gid_type, std::size_t> topology_;

  public:
    enum actions
    {
        discovery_build_network,
        discovery_deploy,
        discovery_topology_lva,
        discovery_empty
    };

    std::vector<naming::id_type> build_network();

    void deploy(std::map<naming::gid_type, std::size_t> const& m)
    { topology_ = m; }

    hpx::uintptr_t topology_lva()
    { return reinterpret_cast<hpx::uintptr_t>(&topology_); } 

    bool empty() 
    { return topology.empty(); } 

    std::size_t operator[] (naming::gid_type const& gid) 
    { return topology[gid]; } 

    typedef actions::result_action0<
        discovery
      , std::vector<naming::id_type>
      , discovery_build_network
      , &discovery::build_network
    > build_network_action;

    typedef actions::action1<
        discovery
      , discovery_deploy
      , std::map<naming::gid_type, std::size_t> const&
      , &discovery::deploy
    > deploy_action;
    
    typedef actions::result_action0<
        discovery
      , hpx::uintptr_t 
      , discovery_topology_lva
      , &discovery::topology_lva
    > topology_lva_action;

    typedef actions::result_action0<
        discovery
      , bool 
      , discovery_empty
      , &discovery::empty
    > empty_action;
};

}}}

#endif // HPX_17581FEA_9F85_45E9_8CB1_36AF08A4809B

