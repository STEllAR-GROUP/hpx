//  Copyright (c) 2007-2010 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/runtime/components/component_factory.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>

#include "server/refcnt.hpp"

///////////////////////////////////////////////////////////////////////////////
// Add factory registration functionality
HPX_REGISTER_COMPONENT_MODULE();
HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    hpx::components::simple_component<hpx::components::refcnt_test::server::refcnt>, 
    refcnt);
HPX_DEFINE_GET_COMPONENT_TYPE(hpx::components::refcnt_test::server::refcnt);

///////////////////////////////////////////////////////////////////////////////
// Serialization support for the refcnt actions
HPX_REGISTER_ACTION_EX(
    hpx::components::refcnt_test::server::refcnt::test_action,
    refcnt_test_action);
