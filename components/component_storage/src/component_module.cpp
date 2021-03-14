//  Copyright (c) 2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/actions_base/basic_action.hpp>
#include <hpx/async_distributed/transfer_continuation_action.hpp>
#include <hpx/components_base/server/component.hpp>
#include <hpx/components_base/server/component_base.hpp>
#include <hpx/runtime_components/component_factory.hpp>

#include <hpx/components/component_storage/server/migrate_to_storage.hpp>

///////////////////////////////////////////////////////////////////////////////
// Add factory registration functionality.
HPX_REGISTER_COMPONENT_MODULE()

typedef hpx::components::server::component_storage component_storage_type;

HPX_REGISTER_COMPONENT(
    hpx::components::component<component_storage_type>,
    component_storage_factory, hpx::components::factory_enabled)
HPX_DEFINE_GET_COMPONENT_TYPE(component_storage_type)

///////////////////////////////////////////////////////////////////////////////
HPX_REGISTER_ACTION(
    hpx::components::server::component_storage::migrate_to_here_action,
    component_storage_migrate_component_to_here_action);
HPX_REGISTER_ACTION(
    hpx::components::server::component_storage::migrate_from_here_action,
    component_storage_migrate_component_from_here_action);
HPX_REGISTER_ACTION(
    hpx::components::server::component_storage::size_action,
    component_storage_size_action);
