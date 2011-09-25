////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/version.hpp>

#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>

#include <hpx/hpx.hpp>
#include <hpx/runtime/components/component_factory.hpp>
#include <hpx/runtime/actions/continuation_impl.hpp>
#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <hpx/runtime/agas/database/backend/stdmap.hpp>
#include <hpx/runtime/agas/network/backend/tcpip.hpp>
#include <hpx/runtime/agas/namespace/symbol.hpp>

using hpx::components::component_agas_symbol_namespace;

typedef hpx::agas::server::symbol_namespace<
    hpx::agas::tag::database::stdmap
  , hpx::agas::tag::network::tcpip
> symbol_namespace_type;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    hpx::components::fixed_component<symbol_namespace_type>,
    stdmap_symbol_namespace);

HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(
    symbol_namespace_type, component_agas_symbol_namespace);

HPX_REGISTER_ACTION_EX(
    symbol_namespace_type::bind_action,
    stdmap_symbol_namespace_bind_action);
HPX_REGISTER_ACTION_EX(
    symbol_namespace_type::resolve_action,
    stdmap_symbol_namespace_resolve_action);
HPX_REGISTER_ACTION_EX(
    symbol_namespace_type::unbind_action,
    stdmap_symbol_namespace_unbind_action);
HPX_REGISTER_ACTION_EX(
    symbol_namespace_type::iterate_action,
    stdmap_symbol_namespace_iterate_action);

