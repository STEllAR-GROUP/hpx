//  Copyright (c) 2011 Bryce Adelstein-Lelbach
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/runtime/components/component_factory_one.hpp>
#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>

#include <tests/correctness/agas/components/server/managed_refcnt_checker.hpp>

HPX_REGISTER_COMPONENT_MODULE();

using hpx::test::server::managed_refcnt_checker;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY_ONE(
    hpx::components::managed_component<managed_refcnt_checker>, 
    managed_refcnt_checker);

HPX_REGISTER_ACTION_EX(
    managed_refcnt_checker::take_reference_action,
    managed_refcnt_checker_take_reference_action);

HPX_DEFINE_GET_COMPONENT_TYPE(managed_refcnt_checker);


