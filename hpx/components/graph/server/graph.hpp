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
            graph_init = 0
        };
        
        // Typedefs for graph
        typedef int count_t;
        typedef long int vertex_t;
        typedef long int edge_t;

        // constructor: initialize graph value
        graph()
          : block_size_(0),
            blocks_(0)
        {}

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        /// Initialize the graph
        void init(count_t order) 
        {            
            std::cout << "Initializng graph of order " << order << "\n";
            
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
            
            // Calculate block distribution
            block_size_ = ceil(order*1.0 / locales.size());
            std::cout << "Block size is " << block_size_ << "\n";
            
            // Build distributed list of vertices
            std::vector<naming::id_type> blocks(locales.size());
            for (int i = 0; i<locales.size(); i++)
            {
                // Allocate remote vector of vertices
                components::memory_block mb(
                    components::memory_block::create(
                        locales[i], sizeof(vertex_t)*block_size_));
                
                // Initialize vector list
                //components::access_memory_block<vertex_t> data(mb.get());
                //std::generate(data.get_ptr(), data.get_ptr()+block_size_, 0);

                // Set gid for remote block
                blocks_[i] = mb.get_gid();

                std::cout << "Allocated memory at locality " << i;
                
                mb.free();
            }
        }

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into an action
        // type, allowing to generate all required boilerplate code for threads,
        // serialization, etc.
        typedef hpx::actions::action1<
            graph, graph_init, count_t, &graph::init
        > init_action;

    private:
        count_t block_size_;
        std::vector<naming::id_type> blocks_;
    };

}}}

#endif
