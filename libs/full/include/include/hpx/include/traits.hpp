//  Copyright (c) 2007-2019 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <hpx/modules/algorithms.hpp>
#include <hpx/modules/async_base.hpp>
#include <hpx/modules/concepts.hpp>
#include <hpx/modules/datastructures.hpp>
#include <hpx/modules/execution_base.hpp>
#include <hpx/modules/functional.hpp>
#include <hpx/modules/futures.hpp>
#include <hpx/modules/iterator_support.hpp>
#include <hpx/modules/plugin.hpp>
#include <hpx/modules/serialization.hpp>
#include <hpx/modules/tag_invoke.hpp>
#include <hpx/modules/timed_execution.hpp>
#include <hpx/modules/type_support.hpp>

#include <hpx/actions_base/traits/action_continuation.hpp>
#include <hpx/actions_base/traits/action_decorate_continuation.hpp>
#include <hpx/actions_base/traits/action_does_termination_detection.hpp>
#include <hpx/actions_base/traits/action_is_target_valid.hpp>
#include <hpx/actions_base/traits/action_priority.hpp>
#include <hpx/actions_base/traits/action_remote_result.hpp>
#include <hpx/actions_base/traits/action_schedule_thread.hpp>
#include <hpx/actions_base/traits/action_select_direct_execution.hpp>
#include <hpx/actions_base/traits/action_stacksize.hpp>
#include <hpx/actions_base/traits/action_was_object_migrated.hpp>
#include <hpx/actions_base/traits/extract_action.hpp>
#include <hpx/actions_base/traits/is_client.hpp>
#include <hpx/actions_base/traits/is_continuation.hpp>
#include <hpx/actions_base/traits/is_distribution_policy.hpp>
#include <hpx/actions_base/traits/is_valid_action.hpp>
#include <hpx/async_distributed/traits/action_trigger_continuation.hpp>
#include <hpx/components_base/traits/action_decorate_function.hpp>
#include <hpx/components_base/traits/component_config_data.hpp>
#include <hpx/components_base/traits/component_heap_type.hpp>
#include <hpx/components_base/traits/component_pin_support.hpp>
#include <hpx/components_base/traits/component_supports_migration.hpp>
#include <hpx/components_base/traits/component_type_database.hpp>
#include <hpx/components_base/traits/component_type_is_compatible.hpp>
#include <hpx/components_base/traits/is_component.hpp>
#include <hpx/components_base/traits/managed_component_policies.hpp>
#include <hpx/parcelset_base/traits/action_get_embedded_parcel.hpp>
#include <hpx/parcelset_base/traits/action_message_handler.hpp>
#include <hpx/parcelset_base/traits/action_serialization_filter.hpp>
