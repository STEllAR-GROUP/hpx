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
#include <hpx/runtime/agas/database/backend/stdmap.hpp>
#include <hpx/runtime/agas/network/backend/tcpip.hpp>

using hpx::lcos::base_lco_with_value;

typedef hpx::agas::server::primary_namespace<
    hpx::agas::tag::database::stdmap,
    hpx::agas::tag::network::tcpip
> agas_component;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    hpx::components::simple_component<agas_component>,
    tcpip_stdmap_primary_namespace);

HPX_DEFINE_GET_COMPONENT_TYPE(agas_component);

HPX_REGISTER_ACTION_EX(
    base_lco_with_value<agas_component::gva_type>::set_result_action,
    set_result_action_agas_tcpip_stdmap_gva_type);
HPX_REGISTER_ACTION_EX(
    base_lco_with_value<agas_component::locality_type>::set_result_action,
    set_result_action_agas_tcpip_stdmap_locality_type);

HPX_REGISTER_ACTION_EX(
    agas_component::bind_locality_action,
    tcpip_stdmap_primary_namespace_bind_locality_action);
HPX_REGISTER_ACTION_EX(
    agas_component::bind_gid_action,
    tcpip_stdmap_primary_namespace_bind_gid_action);
HPX_REGISTER_ACTION_EX(
    agas_component::resolve_locality_action,
    tcpip_stdmap_primary_namespace_resolve_locality_action);
HPX_REGISTER_ACTION_EX(
    agas_component::resolve_gid_action,
    tcpip_stdmap_primary_namespace_resolve_gid_action);
HPX_REGISTER_ACTION_EX(
    agas_component::unbind_action,
    tcpip_stdmap_primary_namespace_unbind_action);

// GID reference count interface
HPX_REGISTER_ACTION_EX(
    agas_component::increment_action,
    tcpip_stdmap_primary_namespace_increment_action);
HPX_REGISTER_ACTION_EX(
    agas_component::decrement_action,
    tcpip_stdmap_primary_namespace_decrement_action);

