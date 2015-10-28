//  Copyright (c) 2013 Shuangyang Yang
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/runtime/components/component_factory.hpp>

#include "server/simple_central_tuplespace.hpp"

//[simple_central_tuplespace_registration_definitions
///////////////////////////////////////////////////////////////////////////////
// Add factory registration functionality.
HPX_REGISTER_COMPONENT_MODULE();

///////////////////////////////////////////////////////////////////////////////
typedef hpx::components::component<
    examples::server::simple_central_tuplespace
> central_tuplespace_type;

HPX_REGISTER_COMPONENT(central_tuplespace_type, simple_central_tuplespace);

///////////////////////////////////////////////////////////////////////////////
// Serialization support for simple_central_tuplespace actions.
HPX_REGISTER_ACTION(
    central_tuplespace_type::wrapped_type::write_action,
    simple_central_tuplespace_write_action);
HPX_REGISTER_ACTION(
    central_tuplespace_type::wrapped_type::read_action,
    simple_central_tuplespace_read_action);
HPX_REGISTER_ACTION(
    central_tuplespace_type::wrapped_type::take_action,
    simple_central_tuplespace_take_action);
//]

