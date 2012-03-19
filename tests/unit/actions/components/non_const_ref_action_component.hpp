//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TEST_UNIT_ACTIONS_NON_CONST_REF_COMPONENT_HPP)
#define HPX_TEST_UNIT_ACTIONS_NON_CONST_REF_COMPONENT_HPP

#include <hpx/include/components.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/util/lightweight_test.hpp>

namespace hpx { namespace test { namespace server
{
    struct non_const_ref_component
      : components::simple_component_base<non_const_ref_component>
    {
        enum actions
        {
            action_non_const_ref_void,
            result_action_non_const_ref,
            direct_action_non_const_ref_void,
            direct_result_action_non_const_ref,
        };

        void non_const_ref_void(int & i)
        {
            HPX_TEST_EQ(i, 9);
        }

        int non_const_ref(int & i)
        {
            return i;
        }

        /*
        typedef hpx::actions::action1<
            non_const_ref_component
          , action_non_const_ref_void
          , int &
          , &non_const_ref_component::non_const_ref_void
        > non_const_ref_void_action;

        typedef hpx::actions::result_action1<
            non_const_ref_component
          , int
          , result_action_non_const_ref
          , int &
          , &non_const_ref_component::non_const_ref
        > non_const_ref_result_action;

        typedef hpx::actions::direct_action1<
            non_const_ref_component
          , direct_action_non_const_ref_void
          , int &
          , &non_const_ref_component::non_const_ref_void
        > non_const_ref_void_direct_action;

        typedef hpx::actions::direct_result_action1<
            non_const_ref_component
          , int
          , direct_result_action_non_const_ref
          , int &
          , &non_const_ref_component::non_const_ref
        > non_const_ref_result_direct_action;
        */
    };

}}}

/*
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::test::server::non_const_ref_component::non_const_ref_void_action
  , non_const_ref_component_non_const_ref_void_action
);
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::test::server::non_const_ref_component::non_const_ref_result_action
  , non_const_ref_component_non_const_ref_result_action
);
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::test::server::non_const_ref_component::non_const_ref_void_direct_action
  , non_const_ref_component_non_const_ref_void_direct_action
);
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::test::server::non_const_ref_component::non_const_ref_result_direct_action
  , non_const_ref_component_non_const_ref_result_direct_action
);
*/

#endif
