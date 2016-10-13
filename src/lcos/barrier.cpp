//  Copyright (c) 2007-2015 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/runtime/components/derived_component_factory.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/runtime/components/runtime_support.hpp>
#include <hpx/lcos/server/barrier.hpp>
#include <hpx/util/serialize_exception.hpp>

///////////////////////////////////////////////////////////////////////////////
// Barrier
typedef hpx::components::managed_component<hpx::lcos::server::barrier> barrier_type;

HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(
    hpx::lcos::server::barrier, hpx::components::component_barrier)
HPX_REGISTER_DERIVED_COMPONENT_FACTORY(barrier_type, barrier,
    "hpx::lcos::base_lco", hpx::components::factory_enabled)

HPX_REGISTER_ACTION_ID(
    hpx::lcos::server::barrier::create_component_action
  , hpx_lcos_server_barrier_create_component_action
  , hpx::actions::hpx_lcos_server_barrier_create_component_action_id
)

