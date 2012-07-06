//  Copyright (c) 2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_226CC70A_D749_4ADA_BB55_70F85566B5CC)
#define HPX_226CC70A_D749_4ADA_BB55_70F85566B5CC

#include <vector>
#include <queue>

#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/util/unlock_lock.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace ag { namespace server
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

        void init(std::size_t item,std::size_t np);
        void compute(std::vector<hpx::naming::id_type> const& point_components);
        double get_item();
        void print();

        // Each of the exposed functions needs to be encapsulated into an
        // action type, generating all required boilerplate code for threads,
        // serialization, etc.

        /// Action codes.
        enum actions
        {
            point_init = 0,
            point_compute = 1,
            point_get_item = 2,
            point_print = 3
        };

        typedef hpx::actions::action2<
            // Component server type.
            point,
            // Action code.
            point_init,
            // Arguments of this action.
            std::size_t,
            std::size_t,
            // Method bound to this action.
            &point::init
        > init_action;

        typedef hpx::actions::action1<
            // Component server type.
            point,
            // Action code.
            point_compute,
            // Arguments of this action.
            std::vector<hpx::naming::id_type> const&,
            // Method bound to this action.
            &point::compute
        > compute_action;

        typedef hpx::actions::action0<
            // Component server type.
            point,
            // Action code.
            point_print,
            // Arguments of this action.
            // Method bound to this action.
            &point::print
        > print_action;

        typedef hpx::actions::result_action0<
            // Component server type.
            point,
            // Return type
            double,
            // Action code.
            point_get_item,
            // Arguments of this action.
            // Method bound to this action.
            &point::get_item
        > get_item_action;

    private:
        //hpx::lcos::local::mutex mtx_;
        hpx::util::spinlock mtx_;
        double value_;
        std::size_t item_;
        std::vector<double> n_;
    };
}}

// Declaration of serialization support for the actions
HPX_REGISTER_ACTION_DECLARATION_EX(
    ag::server::point::init_action,
    ag_point_init_action);

HPX_REGISTER_ACTION_DECLARATION_EX(
    ag::server::point::compute_action,
    ag_point_compute_action);

HPX_REGISTER_ACTION_DECLARATION_EX(
    ag::server::point::get_item_action,
    ag_point_get_item_action);

HPX_REGISTER_ACTION_DECLARATION_EX(
    ag::server::point::print_action,
    ag_point_print_action);
#endif

