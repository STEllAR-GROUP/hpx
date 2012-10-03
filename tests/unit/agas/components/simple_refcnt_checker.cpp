//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/serialization.hpp>

#include <tests/unit/agas/components/server/simple_refcnt_checker.hpp>

HPX_REGISTER_COMPONENT_MODULE();

using hpx::test::server::simple_refcnt_checker;

typedef hpx::components::simple_component<simple_refcnt_checker>
    refcnt_checker_type;

///////////////////////////////////////////////////////////////////////////////
// We use a special component registry for this component as it has to be
// disabled by default. All tests requiring this component to be active will
// enable it explicitly.
HPX_REGISTER_DISABLED_COMPONENT_FACTORY_ONE(
    hpx::components::simple_component<simple_refcnt_checker>,
    simple_refcnt_checker)

///////////////////////////////////////////////////////////////////////////////
HPX_REGISTER_ACTION(
    simple_refcnt_checker::take_reference_action,
    simple_refcnt_checker_take_reference_action);


