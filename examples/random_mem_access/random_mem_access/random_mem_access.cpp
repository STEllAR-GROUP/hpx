//  Copyright (c) 2011 Matt Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/runtime/components/component_factory.hpp>

#include "server/random_mem_access.hpp"

///////////////////////////////////////////////////////////////////////////////
// Add factory registration functionality
HPX_REGISTER_COMPONENT_MODULE();
HPX_REGISTER_COMPONENT(
    hpx::components::component<hpx::components::server::random_mem_access>,
    random_mem_access);

///////////////////////////////////////////////////////////////////////////////
// Serialization support for the random_mem_access actions
HPX_REGISTER_ACTION(
    hpx::components::server::random_mem_access::init_action,
    random_mem_access_init_action);
HPX_REGISTER_ACTION(
    hpx::components::server::random_mem_access::add_action,
    random_mem_access_add_action);
HPX_REGISTER_ACTION(
    hpx::components::server::random_mem_access::query_action,
    random_mem_access_query_action);
HPX_REGISTER_ACTION(
    hpx::components::server::random_mem_access::print_action,
    random_mem_access_print_action);

