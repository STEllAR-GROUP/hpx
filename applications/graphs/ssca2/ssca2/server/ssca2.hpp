//  Copyright (c) 2009-2010 Dylan Stark
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_SERVER_SSCA2_AUG_14_2009_1030AM)
#define HPX_COMPONENTS_SERVER_SSCA2_AUG_14_2009_1030AM

#include <iostream>

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/simple_component_base.hpp>
#include <hpx/components/distributing_factory/distributing_factory.hpp>

#include <hpx/components/graph/graph.hpp>
#include <hpx/components/distributed_set/distributed_set.hpp>

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
            ssca2_large_set = 0,
            ssca2_large_set_local = 1,
            ssca2_extract = 2,
            ssca2_extract_subgraph = 3
        };
        
        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        // This should go somewhere else ... but where?
        struct graph_foo
        {
            graph_foo()
            {}

            graph_foo(naming::id_type const& G)
              : G_(G)
              {}

            naming::id_type G_;

        private:
            // serialization support
            friend class boost::serialization::access;

            template<class Archive>
            void serialize(Archive& ar, const unsigned int)
            {
                ar & G_;
            }
        };

        struct edge
        {
            edge()
            {}

            edge(naming::id_type const& source,
                 naming::id_type const& target,
                 int label)
              : source_(source), target_(target), label_(label)
            {}

            naming::id_type source_;
            naming::id_type target_;
            int label_;

        private:
            // serialization support
            friend class boost::serialization::access;

            template<class Archive>
            void serialize(Archive& ar, const unsigned int)
            {
                ar & source_ & target_ & label_;
            }
        };

        typedef std::vector<edge> edge_set_type;
        typedef distributing_factory::locality_result locality_result;

        typedef distributed_set<edge_set_type> dist_edge_set_type;

        typedef std::vector<graph_foo> graph_set_type;
        typedef distributed_set<graph_set_type> dist_graph_set_type;

        int
        large_set(naming::id_type G,
                  naming::id_type dist_edge_set);

        int
        large_set_local(locality_result local_set,
                        naming::id_type edge_set,
                        naming::id_type local_max_lco,
                        naming::id_type global_max_lco);

        int
        extract(naming::id_type edge_set,
                naming::id_type subgraphs);

        int
        extract_subgraph(naming::id_type H,
                         naming::id_type source,
                         naming::id_type vertex,
                         int d);

        typedef hpx::actions::result_action2<
            ssca2, int, ssca2_large_set,
            naming::id_type, naming::id_type,
            &ssca2::large_set
        > large_set_action;

        typedef hpx::actions::result_action4<
            ssca2, int, ssca2_large_set_local,
            locality_result, naming::id_type, naming::id_type, naming::id_type,
            &ssca2::large_set_local
        > large_set_local_action;

        typedef hpx::actions::result_action2<
            ssca2, int, ssca2_extract,
            naming::id_type, naming::id_type,
            &ssca2::extract
        > extract_action;

        typedef hpx::actions::result_action4<
            ssca2, int, ssca2_extract_subgraph,
            naming::id_type, naming::id_type, naming::id_type, int,
            &ssca2::extract_subgraph
        > extract_subgraph_action;

    };

}}}

#endif
