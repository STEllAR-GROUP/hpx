//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/runtime/components/component_factory_one.hpp>
#include <hpx/runtime/components/test_component_registry.hpp>
#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>

#include <tests/unit/agas/components/server/managed_refcnt_checker.hpp>

HPX_REGISTER_COMPONENT_MODULE();

using hpx::test::server::managed_refcnt_checker;

typedef hpx::components::managed_component<managed_refcnt_checker>
    refcnt_checker_type;

///////////////////////////////////////////////////////////////////////////////
// We use a special component registry for this component as it has to be
// disabled by default. All tests requiring this component to be active will
// enable it explicitly.

HPX_REGISTER_COMPONENT_FACTORY(
    hpx::components::component_factory_one<refcnt_checker_type>,
    managed_refcnt_checker);
HPX_DEF_UNIQUE_COMPONENT_NAME(
    hpx::components::component_factory_one<refcnt_checker_type>,
    managed_refcnt_checker);
template struct hpx::components::component_factory_one<refcnt_checker_type>;
HPX_REGISTER_TEST_COMPONENT_REGISTRY(
    hpx::components::component_factory_one<refcnt_checker_type>,
    managed_refcnt_checker);

///////////////////////////////////////////////////////////////////////////////
HPX_REGISTER_ACTION_EX(
    managed_refcnt_checker::take_reference_action,
    managed_refcnt_checker_take_reference_action);

HPX_DEFINE_GET_COMPONENT_TYPE(managed_refcnt_checker);


