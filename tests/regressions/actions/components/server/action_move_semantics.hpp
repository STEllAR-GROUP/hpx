//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TEST_ACTION_MOVE_SEMANTICS_FEB_23_2012_0937AM)
#define HPX_TEST_ACTION_MOVE_SEMANTICS_FEB_23_2012_0937AM

#include <hpx/include/components.hpp>
#include <hpx/include/actions.hpp>

#include <cstddef>

#include <tests/regressions/actions/components/movable_objects.hpp>

namespace hpx { namespace test { namespace server
{
    struct action_move_semantics
      : components::simple_component_base<action_move_semantics>
    {
        ///////////////////////////////////////////////////////////////////////
        std::size_t test_movable(movable_object const& obj)
        {
            return obj.get_count();
        }

        std::size_t test_non_movable(non_movable_object const& obj)
        {
            return obj.get_count();
        }

        ///////////////////////////////////////////////////////////////////////
        movable_object return_test_movable()
        {
            return movable_object();
        }

        non_movable_object return_test_non_movable()
        {
            return non_movable_object();
        }

        ///////////////////////////////////////////////////////////////////////
        HPX_DEFINE_COMPONENT_ACTION(action_move_semantics,
            test_movable, test_movable_action);
        HPX_DEFINE_COMPONENT_ACTION(action_move_semantics,
            test_non_movable, test_non_movable_action);
        HPX_DEFINE_COMPONENT_DIRECT_ACTION(action_move_semantics,
            test_movable, test_movable_direct_action);
        HPX_DEFINE_COMPONENT_DIRECT_ACTION(action_move_semantics,
            test_non_movable, test_non_movable_direct_action);

        HPX_DEFINE_COMPONENT_ACTION(action_move_semantics,
            return_test_movable, return_test_movable_action);
        HPX_DEFINE_COMPONENT_ACTION(action_move_semantics,
            return_test_non_movable, return_test_non_movable_action);
        HPX_DEFINE_COMPONENT_DIRECT_ACTION(action_move_semantics,
            return_test_movable, return_test_movable_direct_action);
        HPX_DEFINE_COMPONENT_DIRECT_ACTION(action_move_semantics,
            return_test_non_movable, return_test_non_movable_direct_action);
    };
}}}

HPX_REGISTER_ACTION_DECLARATION(
    hpx::test::server::action_move_semantics::test_movable_action,
    action_move_semantics_test_movable_action)
HPX_REGISTER_ACTION_DECLARATION(
    hpx::test::server::action_move_semantics::test_non_movable_action,
    action_move_semantics_test_non_movable_action)
HPX_REGISTER_ACTION_DECLARATION(
    hpx::test::server::action_move_semantics::test_movable_direct_action,
    action_move_semantics_test_movable_direct_action)
HPX_REGISTER_ACTION_DECLARATION(
    hpx::test::server::action_move_semantics::test_non_movable_direct_action,
    action_move_semantics_test_non_movable_direct_action)

HPX_REGISTER_ACTION_DECLARATION(
    hpx::test::server::action_move_semantics::return_test_movable_action,
    action_move_semantics_return_test_movable_action)
HPX_REGISTER_ACTION_DECLARATION(
    hpx::test::server::action_move_semantics::return_test_non_movable_action,
    action_move_semantics_return_test_non_movable_action)
HPX_REGISTER_ACTION_DECLARATION(
    hpx::test::server::action_move_semantics::return_test_movable_direct_action,
    action_move_semantics_return_test_movable_direct_action)
HPX_REGISTER_ACTION_DECLARATION(
    hpx::test::server::action_move_semantics::return_test_non_movable_direct_action,
    action_move_semantics_return_test_non_movable_direct_action)

#endif

