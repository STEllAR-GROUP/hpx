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

#include <hpx/runtime/agas/namespace/component.hpp>
#include <hpx/runtime/agas/database/backend/stdmap.hpp>

typedef hpx::agas::server::component_namespace<
    hpx::agas::tag::database::stdmap
> agas_component;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    hpx::components::simple_component<agas_component>,
    stdmap_component_namespace);

HPX_DEFINE_GET_COMPONENT_TYPE(agas_component);

HPX_REGISTER_ACTION_EX(
    agas_component::bind_prefix_action,
    stdmap_component_namespace_bind_prefix_action);
HPX_REGISTER_ACTION_EX(
    agas_component::bind_name_action,
    stdmap_component_namespace_bind_name_action);
HPX_REGISTER_ACTION_EX(
    agas_component::resolve_id_action,
    stdmap_component_namespace_resolve_id_action);
HPX_REGISTER_ACTION_EX(
    agas_component::resolve_name_action,
    stdmap_component_namespace_resolve_name_action);
HPX_REGISTER_ACTION_EX(
    agas_component::unbind_action,
    stdmap_component_namespace_unbind_action);

