//  Copyright (c) 2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_226CC70A_D748_4ADA_BB55_70F85566B5CC)
#define HPX_226CC70A_D748_4ADA_BB55_70F85566B5CC

#include <vector>

#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/actions/component_action.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace bfs { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    class HPX_COMPONENT_EXPORT point
      : public hpx::components::managed_component_base<point>
    {
    public:
        point()
        {}

        ///////////////////////////////////////////////////////////////////////
        // Exposed functionality of this component.

        /// Initialize the point with the given graph file.  
        void init(std::size_t objectid,std::size_t max_num_neighbors,
            std::string const& graphfile);

        /// Traverse the graph. 
        std::vector<std::size_t> traverse(std::size_t level, std::size_t parent);

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into an
        // action type, generating all required boilerplate code for threads,
        // serialization, etc.

        /// Action codes.
        enum actions
        {
            point_init = 0,
            point_traverse = 1
        };

        typedef hpx::actions::action3<
            // Component server type.
            point,
            // Action code.
            point_init,
            // Arguments of this action.
            std::size_t,
            std::size_t,
            std::string const&,
            // Method bound to this action.
            &point::init
        > init_action;

        typedef hpx::actions::result_action2<
            // Component server type.
            point,
            // Return type.
            std::vector<std::size_t>,
            // Action code.
            point_traverse,
            // Arguments of this action.
            std::size_t,
            std::size_t,
            // Method bound to this action.
            &point::traverse
        > traverse_action;

    private:
        std::size_t idx_;
        std::size_t level_;
        bool visited_;
        std::vector<std::size_t> neighbors_;
        std::size_t parent_;
    };
}}

// Declaration of serialization support for the actions
HPX_REGISTER_ACTION_DECLARATION_EX(
    bfs::server::point::init_action,
    bfs_point_init_action);

HPX_REGISTER_ACTION_DECLARATION_EX(
    bfs::server::point::traverse_action,
    bfs_point_traverse_action);

HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::lcos::base_lco_with_value<std::vector<std::size_t> >::get_value_action,
    get_value_action_vector_size_t);

HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::lcos::base_lco_with_value<std::vector<std::size_t>>::set_result_action,
    set_result_action_vector_size_t);


#endif

