////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/config.hpp>

#include <hpx/hpx.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/runtime/agas/symbol_namespace.hpp>
#include <hpx/runtime/components/component_factory.hpp>

using hpx::components::component_agas_symbol_namespace;

using hpx::agas::server::symbol_namespace;

HPX_REGISTER_COMPONENT(
    hpx::components::fixed_component<symbol_namespace>,
    symbol_namespace, hpx::components::factory_enabled)
HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(
    symbol_namespace, component_agas_symbol_namespace)

HPX_REGISTER_ACTION_ID(
    symbol_namespace::service_action,
    symbol_namespace_service_action,
    hpx::actions::symbol_namespace_service_action_id)

HPX_REGISTER_ACTION_ID(
    symbol_namespace::bulk_service_action,
    symbol_namespace_bulk_service_action,
    hpx::actions::symbol_namespace_bulk_service_action_id)
