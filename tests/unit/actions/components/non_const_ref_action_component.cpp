//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/serialization.hpp>
#include <hpx/runtime/components/test_component_registry.hpp>

#include <tests/unit/actions/components/non_const_ref_action_component.hpp>

HPX_REGISTER_COMPONENT_MODULE();

using hpx::test::server::non_const_ref_component;

typedef hpx::components::simple_component<non_const_ref_component>
    non_const_ref_component_type;

///////////////////////////////////////////////////////////////////////////////
// We use a special component registry for this component as it has to be
// disabled by default. All tests requiring this component to be active will
// enable it explicitly.
HPX_REGISTER_MINIMAL_COMPONENT_FACTORY_EX(
    hpx::components::simple_component<non_const_ref_component>,
    non_const_ref_component,
    hpx::components::factory_disabled)

/*
HPX_REGISTER_ACTION_EX(
    hpx::test::server::non_const_ref_component::non_const_ref_void_action
  , non_const_ref_component_non_const_ref_void_action
);
HPX_REGISTER_ACTION_EX(
    hpx::test::server::non_const_ref_component::non_const_ref_result_action
  , non_const_ref_component_non_const_ref_result_action
);
HPX_REGISTER_ACTION_EX(
    hpx::test::server::non_const_ref_component::non_const_ref_void_direct_action
  , non_const_ref_component_non_const_ref_void_direct_action
);
HPX_REGISTER_ACTION_EX(
    hpx::test::server::non_const_ref_component::non_const_ref_result_direct_action
  , non_const_ref_component_non_const_ref_result_direct_action
);
*/

HPX_DEFINE_GET_COMPONENT_TYPE(non_const_ref_component);

