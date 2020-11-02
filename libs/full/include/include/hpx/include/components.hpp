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

#include <hpx/runtime/components/server/create_component.hpp>
#include <hpx/runtime/components/server/destroy_component.hpp>

#include <hpx/runtime/components/server/invoke_function.hpp>

#include <hpx/runtime/components/client.hpp>
#include <hpx/runtime/components/client_base.hpp>
#include <hpx/runtime/components/stubs/runtime_support.hpp>

#include <hpx/runtime/components/server/component.hpp>
#include <hpx/runtime/components/server/component_base.hpp>

#include <hpx/runtime/components/server/abstract_component_base.hpp>
#include <hpx/runtime/components/server/abstract_migration_support.hpp>
#include <hpx/runtime/components/server/distributed_metadata_base.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/components/server/simple_component_base.hpp>

#include <hpx/runtime/components/server/executor_component.hpp>
#include <hpx/runtime/components/server/locking_hook.hpp>
#include <hpx/runtime/components/server/migration_support.hpp>

#include <hpx/components_base/pinned_ptr.hpp>
#include <hpx/runtime/components/copy_component.hpp>
#include <hpx/runtime/components/migrate_component.hpp>
#include <hpx/runtime/components/new.hpp>

#include <hpx/runtime/components/binpacking_distribution_policy.hpp>
#include <hpx/runtime/components/colocating_distribution_policy.hpp>
#include <hpx/runtime/components/default_distribution_policy.hpp>
#include <hpx/runtime/components/unwrapping_result_policy.hpp>

#include <hpx/runtime/get_ptr.hpp>
#endif
