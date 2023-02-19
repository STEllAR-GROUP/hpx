//  Copyright (c) 2007-2021 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/include/actions.hpp>

#include <hpx/runtime_components/component_factory.hpp>
#include <hpx/runtime_components/component_registry.hpp>
#include <hpx/runtime_components/derived_component_factory.hpp>
#include <hpx/runtime_components/distributed_metadata_base.hpp>
#include <hpx/runtime_components/new.hpp>

#include <hpx/async_colocated/server/destroy_component.hpp>

#include <hpx/actions/invoke_function.hpp>

#include <hpx/components/client.hpp>
#include <hpx/components/client_base.hpp>
#include <hpx/components/executor_component.hpp>
#include <hpx/components/get_ptr.hpp>

#include <hpx/components_base/component_commandline.hpp>
#include <hpx/components_base/component_startup_shutdown.hpp>
#include <hpx/components_base/component_type.hpp>
#include <hpx/components_base/pinned_ptr.hpp>
#include <hpx/components_base/server/abstract_component_base.hpp>
#include <hpx/components_base/server/abstract_migration_support.hpp>
#include <hpx/components_base/server/component.hpp>
#include <hpx/components_base/server/component_base.hpp>
#include <hpx/components_base/server/create_component.hpp>
#include <hpx/components_base/server/locking_hook.hpp>
#include <hpx/components_base/server/managed_component_base.hpp>
#include <hpx/components_base/server/migration_support.hpp>

#include <hpx/runtime_distributed/copy_component.hpp>
#include <hpx/runtime_distributed/migrate_component.hpp>
#include <hpx/runtime_distributed/runtime_support.hpp>
#include <hpx/runtime_distributed/stubs/runtime_support.hpp>

#include <hpx/distribution_policies/binpacking_distribution_policy.hpp>
#include <hpx/distribution_policies/colocating_distribution_policy.hpp>
#include <hpx/distribution_policies/default_distribution_policy.hpp>
#include <hpx/distribution_policies/target_distribution_policy.hpp>
#include <hpx/distribution_policies/unwrapping_result_policy.hpp>
