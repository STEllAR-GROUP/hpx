//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_DISTRIBUTED_RUNTIME)
#include <hpx/include/actions.hpp>

#include <hpx/runtime/components/component_registry.hpp>

#include <hpx/runtime/components/component_factory.hpp>
#include <hpx/runtime/components/derived_component_factory.hpp>

#include <hpx/runtime/components/component_commandline.hpp>
#include <hpx/runtime/components/component_startup_shutdown.hpp>

#include <hpx/components_base/component_type.hpp>

#include <hpx/runtime/components/runtime_support.hpp>

#include <hpx/async_colocated/server/destroy_component.hpp>
#include <hpx/components_base/server/create_component.hpp>

#include <hpx/runtime/components/server/invoke_function.hpp>

#include <hpx/components/client.hpp>
#include <hpx/components/client_base.hpp>
#include <hpx/runtime/components/stubs/runtime_support.hpp>

#include <hpx/components_base/pinned_ptr.hpp>
#include <hpx/components_base/server/abstract_component_base.hpp>
#include <hpx/components_base/server/component.hpp>
#include <hpx/components_base/server/component_base.hpp>
#include <hpx/components_base/server/managed_component_base.hpp>

#include <hpx/runtime/components/server/distributed_metadata_base.hpp>

#include <hpx/components_base/server/abstract_migration_support.hpp>
#include <hpx/components_base/server/locking_hook.hpp>
#include <hpx/components_base/server/migration_support.hpp>
#include <hpx/runtime/components/server/executor_component.hpp>

#include <hpx/runtime_components/copy_component.hpp>
#include <hpx/runtime_components/migrate_component.hpp>
#include <hpx/runtime_components/new.hpp>

#include <hpx/runtime/components/binpacking_distribution_policy.hpp>
#include <hpx/runtime/components/colocating_distribution_policy.hpp>
#include <hpx/runtime/components/default_distribution_policy.hpp>
#include <hpx/runtime/components/unwrapping_result_policy.hpp>

#include <hpx/components/get_ptr.hpp>
#endif
