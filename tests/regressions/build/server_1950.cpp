//  Copyright (c) 2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#define HPX_COMPONENT_EXPORTS

#include "server_1950.hpp"

HPX_REGISTER_COMPONENT_MODULE()

bool test_server::called = false;

typedef hpx::components::simple_component<test_server> server_type;
HPX_REGISTER_COMPONENT(server_type, test_server);

HPX_REGISTER_ACTION(call_action);
