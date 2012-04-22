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
#include <hpx/runtime/agas/component_namespace.hpp>
#include <hpx/runtime/components/component_factory.hpp>
#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

using hpx::components::component_agas_component_namespace;

using hpx::agas::server::component_namespace;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY_EX(
    hpx::components::fixed_component<component_namespace>,
    component_namespace, true)
HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(
    component_namespace, component_agas_component_namespace)

HPX_REGISTER_ACTION_EX(
    component_namespace::service_action,
    component_namespace_service_action)

HPX_REGISTER_ACTION_EX(
    component_namespace::bulk_service_action,
    component_namespace_bulk_service_action)

