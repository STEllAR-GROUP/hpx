////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//  Copyright (c) 2011-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/config.hpp>

#include <hpx/hpx.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/runtime/agas/locality_namespace.hpp>
#include <hpx/runtime/components/component_factory.hpp>

using hpx::components::component_agas_locality_namespace;

using hpx::agas::server::locality_namespace;

HPX_REGISTER_COMPONENT(
    hpx::components::fixed_component<locality_namespace>,
    locality_namespace, hpx::components::factory_enabled)
HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(
    locality_namespace, component_agas_locality_namespace)

HPX_REGISTER_ACTION_ID(
    locality_namespace::service_action,
    locality_namespace_service_action,
    hpx::actions::locality_namespace_service_action_id)

HPX_REGISTER_ACTION_ID(
    locality_namespace::bulk_service_action,
    locality_namespace_bulk_service_action,
    hpx::actions::locality_namespace_bulk_service_action_id)
