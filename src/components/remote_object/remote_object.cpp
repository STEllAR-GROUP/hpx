//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/runtime/components/component_factory.hpp>
#include <hpx/runtime/components/plain_component_factory.hpp>
#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <hpx/components/remote_object/server/remote_object.hpp>
#include <hpx/components/remote_object/new.hpp>

#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>

///////////////////////////////////////////////////////////////////////////////
// Add factory registration functionality
HPX_REGISTER_COMPONENT_MODULE()

typedef hpx::components::server::remote_object remote_object_type;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY_EX(
    hpx::components::managed_component<remote_object_type>,
    remote_object, hpx::components::factory_enabled)
HPX_DEFINE_GET_COMPONENT_TYPE(remote_object_type)

///////////////////////////////////////////////////////////////////////////////
// Serialization support for the remote_object actions
/*
HPX_REGISTER_ACTION(
    hpx::components::server::remote_object_apply_action<void>,
    remote_object_apply_action_void);
*/
HPX_REGISTER_ACTION(
    remote_object_type::set_dtor_action,
    remote_object_set_dtor_action)

HPX_REGISTER_PLAIN_ACTION_EX(
    hpx::components::remote_object::new_impl_action,
    remote_object_new_impl_action)
