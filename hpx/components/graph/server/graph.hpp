//  Copyright (c) 2007-2009 Dylan Stark
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_SERVER_GRAPH_MAY_17_2008_0731PM)
#define HPX_COMPONENTS_SERVER_GRAPH_MAY_17_2008_0731PM

#include <iostream>

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/actions/component_action.hpp>

#include <hpx/components/vertex/vertex.hpp>
#include <hpx/components/vertex_list/vertex_list.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server 
{
    ///////////////////////////////////////////////////////////////////////////
    /// The graph is an HPX component. 
    ///
    class graph 
      : public components::detail::managed_component_base<graph> 
    {
    public:
        // parcel action code: the action to be performed on the destination 
        // object (the graph)
        enum actions
        {
            graph_init = 0,
            graph_order = 1,
            graph_size = 2,
            graph_add_edge = 3,
            graph_vertex_name = 4
        };
        
        // Typedefs for graph
        typedef int count_t;
        typedef long int vertex_t;
        typedef long int edge_t;

        // constructor: initialize graph value
        graph()
          : block_size_(0),
            blocks_(0),
            size_(0)
        {
            applier::applier& appl = applier::get_applier();

            using hpx::components::vertex_list;
            vertex_list vertices_(vertex_list::create(appl.get_runtime_support_gid()));
        }

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        /// Initialize the graph
        int init(count_t order) 
        {            
            std::cout << "Initializing graph of order " << order << std::endl;
            
            // Get list of all known localities
            std::vector<naming::id_type> locales;
            naming::id_type locale;
            applier::applier& appl = applier::get_applier();
            if (appl.get_remote_prefixes(locales))
            {
                locale = locales[0];
            }
            else
            {
                locale = appl.get_runtime_support_gid();
            }
            locales.push_back(appl.get_runtime_support_gid());
            
            // Create a vertex_list and initialize
            components::component_type vertex_type = components::get_component_type<vertex>();
            vertices_.init(vertex_type, order);

            return 0;
        }

        int order(void)
        {
            return vertices_.size();
        }

        int size(void)
        {
            return size_;
        }

        int add_edge(naming::id_type u_g, naming::id_type v_g, int label)
        {
            std::cout << "Adding edge ("
                      << hpx::components::stubs::vertex::label(u_g) << ", "
                      << hpx::components::stubs::vertex::label(v_g) << ", "
                      << label << ")" << std::endl;

            // Extend vertex components for adding edges

        	++size_;

        	return 0;
        }

        naming::id_type vertex_name(int id)
        {
        	return vertices_.at_index(id);
        }

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into an action
        // type, allowing to generate all required boilerplate code for threads,
        // serialization, etc.
        typedef hpx::actions::result_action1<
            graph, int, graph_init, count_t, &graph::init
        > init_action;

        typedef hpx::actions::result_action0<
            graph, int, graph_order, &graph::order
        > order_action;

        typedef hpx::actions::result_action0<
            graph, int, graph_size, &graph::size
        > size_action;

        typedef hpx::actions::result_action3<
			graph, int, graph_add_edge, naming::id_type, naming::id_type, int, &graph::add_edge
		> add_edge_action;

        typedef hpx::actions::result_action1<
			graph, naming::id_type, graph_vertex_name, int, &graph::vertex_name
		> vertex_name_action;

    private:
        count_t block_size_;
        std::vector<naming::id_type> blocks_;

        int size_;

        vertex_list vertices_;
    };

}}}

#endif
