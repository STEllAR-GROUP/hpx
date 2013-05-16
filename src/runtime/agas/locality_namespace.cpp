////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//  Copyright (c) 2011-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>

#include <hpx/hpx.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/runtime/agas/locality_namespace.hpp>
#include <hpx/runtime/components/component_factory.hpp>
#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

using hpx::components::component_agas_locality_namespace;

using hpx::agas::server::locality_namespace;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    hpx::components::fixed_component<locality_namespace>,
    locality_namespace, hpx::components::factory_enabled)
HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(
    locality_namespace, component_agas_locality_namespace)

HPX_REGISTER_ACTION(
    locality_namespace::service_action,
    locality_namespace_service_action)

HPX_REGISTER_ACTION(
    locality_namespace::bulk_service_action,
    locality_namespace_bulk_service_action)
