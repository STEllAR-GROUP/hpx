////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/optional.hpp>

#include <hpx/hpx.hpp>
#include <hpx/runtime/components/component_factory.hpp>
#include <hpx/runtime/actions/continuation_impl.hpp>
#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>
#include <hpx/lcos/base_lco.hpp>

#include <hpx/runtime/agas/database/backend/stdmap.hpp>
#include <hpx/runtime/agas/network/backend/tcpip.hpp>
#include <hpx/runtime/agas/namespace/primary.hpp>

using boost::optional;

using hpx::components::component_agas_symbol_namespace;
using hpx::components::component_base_lco_with_value;
using hpx::lcos::base_lco_with_value;
using hpx::naming::gid_type;

typedef hpx::agas::server::primary_namespace<
    hpx::agas::tag::database::stdmap,
    hpx::agas::tag::network::tcpip
> agas_component;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    hpx::components::fixed_component<agas_component>,
    stdmap_tcpip_primary_namespace);

HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(
    agas_component, component_agas_primary_namespace);
HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(
    agas_component::base_type, component_agas_primary_namespace);

HPX_REGISTER_ACTION_EX(
    base_lco_with_value<agas_component::gva_type>::set_result_action,
    set_result_action_agas_stdmap_tcpip_gva_type);
HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(
    base_lco_with_value<agas_component::gva_type>,
    component_base_lco_with_value);

HPX_REGISTER_ACTION_EX(
    base_lco_with_value<agas_component::locality_type>::set_result_action,
    set_result_action_agas_stdmap_tcpip_locality_type);
HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(
    base_lco_with_value<agas_component::locality_type>,
    component_base_lco_with_value);

HPX_REGISTER_ACTION_EX(
    base_lco_with_value<optional<agas_component::gva_type> >::set_result_action,
    set_result_action_agas_stdmap_tcpip_binding_type);
HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(
    base_lco_with_value<optional<agas_component::gva_type> >,
    component_base_lco_with_value);

HPX_REGISTER_ACTION_EX(
    agas_component::bind_locality_action,
    stdmap_tcpip_primary_namespace_bind_locality_action);
HPX_REGISTER_ACTION_EX(
    agas_component::bind_gid_action,
    stdmap_tcpip_primary_namespace_bind_gid_action);
HPX_REGISTER_ACTION_EX(
    agas_component::resolve_locality_action,
    stdmap_tcpip_primary_namespace_resolve_locality_action);
HPX_REGISTER_ACTION_EX(
    agas_component::resolve_gid_action,
    stdmap_tcpip_primary_namespace_resolve_gid_action);
HPX_REGISTER_ACTION_EX(
    agas_component::unbind_action,
    stdmap_tcpip_primary_namespace_unbind_action);
HPX_REGISTER_ACTION_EX(
    agas_component::localities_action,
    stdmap_tcpip_primary_namespace_localities_action);

// GID reference count interface
HPX_REGISTER_ACTION_EX(
    agas_component::increment_action,
    stdmap_tcpip_primary_namespace_increment_action);
HPX_REGISTER_ACTION_EX(
    agas_component::decrement_action,
    stdmap_tcpip_primary_namespace_decrement_action);

