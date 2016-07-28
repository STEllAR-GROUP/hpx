////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/config.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/runtime/agas/detail/local_primary_namespace.hpp>
#include <hpx/runtime/agas/server/primary_namespace.hpp>
#include <hpx/runtime/components/component_factory.hpp>

using hpx::components::component_agas_primary_namespace;

using hpx::agas::server::primary_namespace;

HPX_REGISTER_COMPONENT(
    hpx::components::fixed_component<primary_namespace>,
    primary_namespace, hpx::components::factory_enabled)
HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(
    hpx::agas::detail::local_primary_namespace, component_agas_primary_namespace)

HPX_REGISTER_COMPONENT(
    hpx::components::fixed_component<hpx::agas::detail::local_primary_namespace>,
    local_primary_namespace, hpx::components::factory_enabled)
HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(
    hpx::agas::server::primary_namespace, component_agas_primary_namespace)

HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(
    hpx::agas::detail::primary_namespace_base, component_agas_primary_namespace)

HPX_REGISTER_ACTION_ID(
    hpx::agas::detail::primary_namespace_base::service_action,
    primary_namespace_service_action,
    hpx::actions::primary_namespace_service_action_id)

HPX_REGISTER_ACTION_ID(
    hpx::agas::detail::primary_namespace_base::bulk_service_action,
    primary_namespace_bulk_service_action,
    hpx::actions::primary_namespace_bulk_service_action_id)

HPX_REGISTER_ACTION_ID(
    hpx::agas::detail::primary_namespace_base::route_action,
    primary_namespace_route_action,
    hpx::actions::primary_namespace_route_action_id)
