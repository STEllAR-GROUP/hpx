////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>

#include <hpx/hpx.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/runtime/agas/primary_namespace.hpp>
#include <hpx/runtime/components/component_factory.hpp>
#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

using hpx::components::component_agas_primary_namespace;

using hpx::agas::server::primary_namespace;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY_EX(
    hpx::components::fixed_component<primary_namespace>,
    primary_namespace, hpx::components::factory_enabled)
HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(
    primary_namespace, component_agas_primary_namespace)

HPX_REGISTER_ACTION(
    primary_namespace::service_action,
    primary_namespace_service_action)

HPX_REGISTER_ACTION(
    primary_namespace::bulk_service_action,
    primary_namespace_bulk_service_action)

HPX_REGISTER_ACTION(
    primary_namespace::route_action,
    primary_namespace_route_action)
