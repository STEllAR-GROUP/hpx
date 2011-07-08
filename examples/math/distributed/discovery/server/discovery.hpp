////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_17581FEA_9F85_45E9_8CB1_36AF08A4809B)
#define HPX_17581FEA_9F85_45E9_8CB1_36AF08A4809B

#include <vector>

#include <hpx/config.hpp>
#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/server/simple_component_base.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/util/uintptr_t.hpp>

namespace hpx { namespace balancing
{

typedef std::map<boost::uint32_t, boost::uint32_t> topology_map;

namespace server
{

struct HPX_COMPONENT_EXPORT discovery
    : components::simple_component_base<discovery> 
{
    typedef components::simple_component_base<discovery> base_type; 
    
    static boost::uint32_t report_shepherd_count();

  private:
    topology_map topology_;
    boost::uint32_t total_shepherds_;

  public:
    enum actions
    {
        discovery_build_network,
        discovery_deploy,
        discovery_topology_lva,
        discovery_total_shepherds,
        discovery_empty
    };

    std::vector<naming::id_type> build_network();

    void deploy(topology_map const& m, boost::uint32_t total_shepherds)
    {
        topology_ = m;
        total_shepherds_ = total_shepherds;
    }

    hpx::uintptr_t topology_lva()
    { return reinterpret_cast<hpx::uintptr_t>(&topology_); } 

    boost::uint32_t total_shepherds()
    { return total_shepherds_; }

    bool empty() 
    { return topology_.empty(); } 
    
    boost::uint32_t operator[] (naming::id_type const& gid) 
    { return topology_[naming::get_prefix_from_gid(gid.get_gid())]; } 

    boost::uint32_t operator[] (naming::gid_type const& gid) 
    { return topology_[naming::get_prefix_from_gid(gid)]; } 

    boost::uint32_t operator[] (boost::uint32_t const& prefix) 
    { return topology_[prefix]; } 

    typedef actions::result_action0<
        discovery
      , std::vector<naming::id_type>
      , discovery_build_network
      , &discovery::build_network
    > build_network_action;

    typedef actions::action2<
        discovery
      , discovery_deploy
      , topology_map const& 
      , boost::uint32_t 
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
      , boost::uint32_t 
      , discovery_total_shepherds
      , &discovery::total_shepherds
    > total_shepherds_action;

    typedef actions::result_action0<
        discovery
      , bool 
      , discovery_empty
      , &discovery::empty
    > empty_action;
};

}}}

#endif // HPX_17581FEA_9F85_45E9_8CB1_36AF08A4809B

