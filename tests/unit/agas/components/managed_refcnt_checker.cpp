//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/serialization.hpp>

#include <tests/unit/agas/components/server/managed_refcnt_checker.hpp>

HPX_REGISTER_COMPONENT_MODULE();

using hpx::test::server::managed_refcnt_checker;

typedef hpx::components::managed_component<managed_refcnt_checker>
    refcnt_checker_type;

///////////////////////////////////////////////////////////////////////////////
// We use a special component registry for this component as it has to be
// disabled by default. All tests requiring this component to be active will
// enable it explicitly.
HPX_REGISTER_MINIMAL_COMPONENT_FACTORY_ONE_EX(
    hpx::components::managed_component<managed_refcnt_checker>,
    managed_refcnt_checker,
    hpx::components::factory_disabled)

///////////////////////////////////////////////////////////////////////////////
HPX_REGISTER_ACTION_EX(
    managed_refcnt_checker::take_reference_action,
    managed_refcnt_checker_take_reference_action);

HPX_DEFINE_GET_COMPONENT_TYPE(managed_refcnt_checker);

