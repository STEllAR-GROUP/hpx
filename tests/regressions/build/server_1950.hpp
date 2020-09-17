//  Copyright (c) 2014 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/include/components.hpp>
#include <hpx/include/actions.hpp>

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
