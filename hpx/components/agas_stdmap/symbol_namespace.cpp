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

#include <hpx/runtime/agas/namespace/symbol.hpp>
#include <hpx/runtime/agas/database/backend/stdmap.hpp>

using hpx::components::component_agas_symbol_namespace;

typedef hpx::agas::server::symbol_namespace<
    hpx::agas::tag::database::stdmap
> agas_component;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    hpx::components::simple_component<agas_component>,
    stdmap_symbol_namespace);

HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(
    agas_component, component_agas_symbol_namespace);

HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(
    agas_component::base_type, component_agas_symbol_namespace);

HPX_REGISTER_ACTION_EX(
    agas_component::bind_action,
    stdmap_symbol_namespace_bind_action);
HPX_REGISTER_ACTION_EX(
    agas_component::rebind_action,
    stdmap_symbol_namespace_rebind_action);
HPX_REGISTER_ACTION_EX(
    agas_component::resolve_action,
    stdmap_symbol_namespace_resolve_action);
HPX_REGISTER_ACTION_EX(
    agas_component::unbind_action,
    stdmap_symbol_namespace_unbind_action);

