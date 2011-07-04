//  Copyright (c) 2011 Matt Anderson
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/runtime/components/component_factory.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>

#include "server/contact.hpp"

///////////////////////////////////////////////////////////////////////////////
// Add factory registration functionality
HPX_REGISTER_COMPONENT_MODULE();
HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    hpx::components::simple_component<hpx::components::server::contact>, 
    contact);

///////////////////////////////////////////////////////////////////////////////
// Serialization support for the contact actions
HPX_REGISTER_ACTION_EX(
    hpx::components::server::contact::init_action,
    contact_init_action);
HPX_REGISTER_ACTION_EX(
    hpx::components::server::contact::contactsearch_action,
    contact_contactsearch_action);
HPX_REGISTER_ACTION_EX(
    hpx::components::server::contact::query_action,
    contact_query_action);
HPX_REGISTER_ACTION_EX(
    hpx::components::server::contact::contactenforce_action,
    contact_contactenforce_action);
HPX_DEFINE_GET_COMPONENT_TYPE(hpx::components::server::contact);

