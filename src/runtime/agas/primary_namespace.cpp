////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>

#include <hpx/hpx.hpp>
#include <hpx/runtime/components/component_factory.hpp>
#include <hpx/runtime/actions/continuation_impl.hpp>
#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>
#include <hpx/lcos/base_lco.hpp>

#include <hpx/runtime/agas/namespace/primary.hpp>
#include <hpx/runtime/agas/database/backend/default.hpp>
#include <hpx/runtime/agas/network/backend/default.hpp>

typedef hpx::agas::server::primary_namespace<
    hpx::agas::tag::database::default_,
    hpx::agas::tag::network::default_
> agas_component;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    hpx::components::simple_component<agas_component>,
    primary_namespace);

HPX_DEFINE_GET_COMPONENT_TYPE(agas_component);

// gva_type is protocol specific, so register it here instead of in
// common_actions.cpp.
HPX_REGISTER_ACTION_EX(
    hpx::lcos::base_lco_with_value<agas_component::gva_type>::set_result_action,
    set_result_action_agas_default_gva_type);

HPX_REGISTER_ACTION_EX(
    agas_component::bind_locality_action,
    primary_namespace_bind_locality_action);
HPX_REGISTER_ACTION_EX(
    agas_component::bind_gid_action,
    primary_namespace_bind_gid_action);
HPX_REGISTER_ACTION_EX(
    agas_component::resolve_locality_action,
    primary_namespace_resolve_locality_action);
HPX_REGISTER_ACTION_EX(
    agas_component::resolve_gid_action,
    primary_namespace_resolve_gid_action);
HPX_REGISTER_ACTION_EX(
    agas_component::unbind_action,
    primary_namespace_unbind_action);

// GID reference count interface
HPX_REGISTER_ACTION_EX(
    agas_component::increment_action,
    primary_namespace_increment_action);
HPX_REGISTER_ACTION_EX(
    agas_component::decrement_action,
    primary_namespace_decrement_action);

