//  Copyright (c) 2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_226CC70A_D748_4ADA_BB55_70F85566B5CC)
#define HPX_226CC70A_D748_4ADA_BB55_70F85566B5CC

#include <vector>

#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include "graph_generator.h"
#include "splittable_mrg.h"

///////////////////////////////////////////////////////////////////////////////
namespace graph500 { namespace server
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

        // kernel 1
        void init(std::size_t objectid,std::size_t scale,std::size_t number_partitions);

        // Each of the exposed functions needs to be encapsulated into an
        // action type, generating all required boilerplate code for threads,
        // serialization, etc.

        /// Action codes.
        enum actions
        {
            point_init = 0
        };

        typedef hpx::actions::action3<
            // Component server type.
            point,
            // Action code.
            point_init,
            // Arguments of this action.
            std::size_t,
            std::size_t,
            std::size_t,
            // Method bound to this action.
            &point::init
        > init_action;

    private:
        hpx::lcos::local_mutex mtx_;
        std::size_t idx_;
    };
}}

// Declaration of serialization support for the actions
HPX_REGISTER_ACTION_DECLARATION_EX(
    graph500::server::point::init_action,
    graph500_point_init_action);

HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::lcos::base_lco_with_value<std::vector<std::size_t> >::get_value_action,
    get_value_action_vector_size_t);

HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::lcos::base_lco_with_value<std::vector<std::size_t>>::set_result_action,
    set_result_action_vector_size_t);


#endif

