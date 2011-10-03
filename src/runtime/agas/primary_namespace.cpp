////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>
//#include <boost/serialization/optional.hpp>

#include <hpx/hpx.hpp>
#include <hpx/runtime/components/component_factory.hpp>
#include <hpx/runtime/actions/continuation_impl.hpp>
#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>
#include <hpx/lcos/base_lco.hpp>
#include <hpx/runtime/agas/namespace/primary.hpp>

using hpx::components::component_agas_primary_namespace;

using hpx::agas::server::primary_namespace;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY_EX(
    hpx::components::fixed_component<primary_namespace>,
    primary_namespace, true);

HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(
    primary_namespace, component_agas_primary_namespace);

HPX_REGISTER_ACTION_EX(
    primary_namespace::bind_locality_action,
    primary_namespace_bind_locality_action);
HPX_REGISTER_ACTION_EX(
    primary_namespace::bind_gid_action,
    primary_namespace_bind_gid_action);
HPX_REGISTER_ACTION_EX(
    primary_namespace::page_fault_action,
    primary_namespace_page_fault_action);
HPX_REGISTER_ACTION_EX(
    primary_namespace::unbind_locality_action,
    primary_namespace_unbind_locality_action);
HPX_REGISTER_ACTION_EX(
    primary_namespace::unbind_gid_action,
    primary_namespace_unbind_gid_action);
HPX_REGISTER_ACTION_EX(
    primary_namespace::localities_action,
    primary_namespace_localities_action);
HPX_REGISTER_ACTION_EX(
    primary_namespace::increment_action,
    primary_namespace_increment_action);
HPX_REGISTER_ACTION_EX(
    primary_namespace::decrement_action,
    primary_namespace_decrement_action);

