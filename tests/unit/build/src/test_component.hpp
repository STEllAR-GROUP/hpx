//  Copyright (c) 2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_TESTS_UNIT_BUILD_TEST_COMPONENT_HPP
#define HPX_TESTS_UNIT_BUILD_TEST_COMPONENT_HPP

#include <hpx/include/components.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/util/lightweight_test.hpp>

struct HPX_COMPONENT_EXPORT test_server
  : hpx::components::simple_component_base<test_server>
{
    test_server() {}

    void call() { called = true; }
    HPX_DEFINE_COMPONENT_ACTION(test_server, call);

    static bool called;
};

typedef test_server::call_action call_action;
HPX_REGISTER_ACTION_DECLARATION(call_action);

#endif
