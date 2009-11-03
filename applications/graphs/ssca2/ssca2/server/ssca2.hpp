//  Copyright (c) 2009-2010 Dylan Stark
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_SERVER_SSCA2_AUG_14_2009_1030AM)
#define HPX_COMPONENTS_SERVER_SSCA2_AUG_14_2009_1030AM

#include <iostream>

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/simple_component_base.hpp>
#include <hpx/components/distributing_factory/distributing_factory.hpp>

#include <hpx/components/graph/graph.hpp>
#include <hpx/components/distributed_set/distributed_set.hpp>

#include <hpx/components/distributed_map/distributed_map.hpp>
#include <hpx/components/distributed_map/local_map.hpp>

#include "boost/serialization/map.hpp"

#include "../../pbreak/pbreak.hpp"

#include <hpx/lcos/mutex.hpp>
#include <hpx/util/spinlock_pool.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    /// The ssca2 is an HPX component.
    ///
    class HPX_COMPONENT_EXPORT ssca2
      : public simple_component_base<ssca2>
    {
    private:
        typedef simple_component_base<ssca2> base_type;
        
    public:
        ssca2();
        
        typedef hpx::components::server::ssca2 wrapping_type;
        
        enum actions
        {
            ssca2_read_graph = 0,
            ssca2_large_set = 1,
            ssca2_large_set_local = 2,
            ssca2_init_props_map = 6,
            ssca2_init_props_map_local = 7
        };
        
        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        typedef distributing_factory::locality_result locality_result;

        typedef std::map<naming::id_type,naming::id_type> gids_map_type;
        typedef distributed_map<gids_map_type> dist_gids_map_type;
        typedef local_map<gids_map_type> local_gids_map_type;

        int
        read_graph(naming::id_type G,
                   std::string filename);

        int
        large_set(naming::id_type G,
                  naming::id_type dist_edge_set);

        int
        large_set_local(naming::id_type local_set,
                        naming::id_type edge_set,
                        naming::id_type local_max_lco,
                        naming::id_type global_max_lco);

        int
        init_props_map(naming::id_type P,
                       naming::id_type G);

        int
        init_props_map_local(naming::id_type local_props,
                             locality_result local_vertices);

        typedef hpx::actions::result_action2<
            ssca2, int, ssca2_read_graph,
            naming::id_type, std::string,
            &ssca2::read_graph
        > read_graph_action;

        typedef hpx::actions::result_action2<
            ssca2, int, ssca2_large_set,
            naming::id_type, naming::id_type,
            &ssca2::large_set
        > large_set_action;

        typedef hpx::actions::result_action4<
            ssca2, int, ssca2_large_set_local,
            naming::id_type, naming::id_type, naming::id_type, naming::id_type,
            &ssca2::large_set_local
        > large_set_local_action;

        typedef hpx::actions::result_action2<
            ssca2, int, ssca2_init_props_map,
            naming::id_type, naming::id_type,
            &ssca2::init_props_map
        > init_props_map_action;

        typedef hpx::actions::result_action2<
            ssca2, int, ssca2_init_props_map_local,
            naming::id_type, locality_result,
            &ssca2::init_props_map_local
        > init_props_map_local_action;

    };

}}}

#endif
