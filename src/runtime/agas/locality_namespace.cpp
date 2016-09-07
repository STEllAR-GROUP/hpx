////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//  Copyright (c) 2011-2013 Hartmut Kaiser
//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/config.hpp>
#include <hpx/lcos/base_lco_with_value.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/runtime/agas/locality_namespace.hpp>
#include <hpx/runtime/agas/server/locality_namespace.hpp>
#include <hpx/runtime/components/component_factory.hpp>

using hpx::components::component_agas_locality_namespace;

using hpx::agas::server::locality_namespace;

HPX_REGISTER_COMPONENT(
    hpx::components::fixed_component<locality_namespace>,
    locality_namespace, hpx::components::factory_enabled)
HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(
    locality_namespace, component_agas_locality_namespace)

HPX_REGISTER_ACTION_ID(
    locality_namespace::allocate_action,
    locality_namespace_allocate_action,
    hpx::actions::locality_namespace_allocate_action_id)

HPX_REGISTER_ACTION_ID(
    locality_namespace::free_action,
    locality_namespace_free_action,
    hpx::actions::locality_namespace_free_action_id)

HPX_REGISTER_ACTION_ID(
    locality_namespace::localities_action,
    locality_namespace_localities_action,
    hpx::actions::locality_namespace_localities_action_id)

HPX_REGISTER_ACTION_ID(
    locality_namespace::resolve_locality_action,
    locality_namespace_resolve_locality_action,
    hpx::actions::locality_namespace_resolve_locality_action_id)

HPX_REGISTER_ACTION_ID(
    locality_namespace::get_num_localities_action,
    locality_namespace_get_num_localities_action,
    hpx::actions::locality_namespace_get_num_localities_action_id)

HPX_REGISTER_ACTION_ID(
    locality_namespace::get_num_threads_action,
    locality_namespace_get_num_threads_action,
    hpx::actions::locality_namespace_get_num_threads_action_id)

HPX_REGISTER_ACTION_ID(
    locality_namespace::get_num_overall_threads_action,
    locality_namespace_get_num_overall_threads_action,
    hpx::actions::locality_namespace_get_num_overall_threads_action_id)

HPX_REGISTER_ACTION_ID(
    locality_namespace::statistics_counter_action,
    locality_namespace_statistics_counter_action,
    hpx::actions::locality_namespace_statistics_counter_action_id)

namespace hpx { namespace agas {
    locality_namespace::~locality_namespace()
    {}
}}
