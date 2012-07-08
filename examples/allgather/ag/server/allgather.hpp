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
    class HPX_COMPONENT_EXPORT allgather
      : public hpx::components::managed_component_base<allgather>
    {
    public:
        allgather()
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
            allgather_init = 0,
            allgather_compute = 1,
            allgather_get_item = 2,
            allgather_print = 3
        };

        typedef hpx::actions::action2<
            // Component server type.
            allgather,
            // Action code.
            allgather_init,
            // Arguments of this action.
            std::size_t,
            std::size_t,
            // Method bound to this action.
            &allgather::init
        > init_action;

        typedef hpx::actions::action1<
            // Component server type.
            allgather,
            // Action code.
            allgather_compute,
            // Arguments of this action.
            std::vector<hpx::naming::id_type> const&,
            // Method bound to this action.
            &allgather::compute
        > compute_action;

        typedef hpx::actions::action0<
            // Component server type.
            allgather,
            // Action code.
            allgather_print,
            // Arguments of this action.
            // Method bound to this action.
            &allgather::print
        > print_action;

        typedef hpx::actions::result_action0<
            // Component server type.
            allgather,
            // Return type
            double,
            // Action code.
            allgather_get_item,
            // Arguments of this action.
            // Method bound to this action.
            &allgather::get_item
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
    ag::server::allgather::init_action,
    allgather_init_action);

HPX_REGISTER_ACTION_DECLARATION_EX(
    ag::server::allgather::compute_action,
    allgather_compute_action);

HPX_REGISTER_ACTION_DECLARATION_EX(
    ag::server::allgather::get_item_action,
    allgather_get_item_action);

HPX_REGISTER_ACTION_DECLARATION_EX(
    ag::server::allgather::print_action,
    allgather_print_action);
#endif

