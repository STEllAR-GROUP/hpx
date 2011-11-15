//  Copyright (c) 2011 Matt Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/components/component_factory.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>

#include "server/element.hpp"

///////////////////////////////////////////////////////////////////////////////
// Add factory registration functionality
HPX_REGISTER_COMPONENT_MODULE();
HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    hpx::components::managed_component<random_mem_access::server::element>,
    random_mem_access_element);

///////////////////////////////////////////////////////////////////////////////
// Serialization support for the random_mem_access::element actions
HPX_REGISTER_ACTION_EX(
    random_mem_access::server::element::init_action,
    random_mem_access_element_init_action);
HPX_REGISTER_ACTION_EX(
    random_mem_access::server::element::add_action,
    random_mem_access_element_add_action);
HPX_REGISTER_ACTION_EX(
    random_mem_access::server::element::print_action,
    random_mem_access_element_print_action);
HPX_DEFINE_GET_COMPONENT_TYPE(random_mem_access::server::element);

