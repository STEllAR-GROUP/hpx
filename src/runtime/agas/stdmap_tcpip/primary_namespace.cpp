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

using hpx::components::component_agas_symbol_namespace;

typedef hpx::agas::server::primary_namespace<
    hpx::agas::tag::database::stdmap
  , hpx::agas::tag::network::tcpip
> agas_component;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    hpx::components::fixed_component<agas_component>,
    stdmap_tcpip_primary_namespace);

HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(
    agas_component, component_agas_primary_namespace);

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
HPX_REGISTER_ACTION_EX(
    agas_component::increment_action,
    stdmap_tcpip_primary_namespace_increment_action);
HPX_REGISTER_ACTION_EX(
    agas_component::decrement_action,
    stdmap_tcpip_primary_namespace_decrement_action);

