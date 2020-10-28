//  Copyright (c) 2007-2019 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/algorithms/traits/pointer_category.hpp>
#include <hpx/algorithms/traits/segmented_iterator_traits.hpp>
#include <hpx/async_base/traits/is_launch_policy.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/concepts/has_member_xxx.hpp>
#include <hpx/concepts/has_xxx.hpp>
#include <hpx/datastructures/traits/is_tuple_like.hpp>
#include <hpx/datastructures/traits/supports_streaming_with_any.hpp>
#include <hpx/execution/traits/is_executor.hpp>
#include <hpx/execution/traits/is_executor_parameters.hpp>
#include <hpx/functional/traits/get_function_address.hpp>
#include <hpx/functional/traits/get_function_annotation.hpp>
#include <hpx/functional/traits/is_bind_expression.hpp>
#include <hpx/functional/traits/is_invocable.hpp>
#include <hpx/functional/traits/is_placeholder.hpp>
#include <hpx/futures/traits/acquire_future.hpp>
#include <hpx/futures/traits/acquire_shared_state.hpp>
#include <hpx/futures/traits/future_access.hpp>
#include <hpx/futures/traits/future_traits.hpp>
#include <hpx/futures/traits/is_future.hpp>
#include <hpx/futures/traits/is_future_range.hpp>
#include <hpx/futures/traits/is_future_tuple.hpp>
#include <hpx/futures/traits/promise_local_result.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/iterator_support/traits/is_range.hpp>
#include <hpx/plugin/traits/plugin_config_data.hpp>
#include <hpx/serialization/traits/brace_initializable_traits.hpp>
#include <hpx/serialization/traits/is_bitwise_serializable.hpp>
#include <hpx/serialization/traits/needs_automatic_registration.hpp>
#include <hpx/serialization/traits/polymorphic_traits.hpp>
#include <hpx/serialization/traits/serialization_access_data.hpp>
#include <hpx/timed_execution/traits/is_timed_executor.hpp>
#include <hpx/traits/is_continuation.hpp>
#include <hpx/type_support/detail/wrap_int.hpp>

#if defined(HPX_HAVE_DISTRIBUTED_RUNTIME)
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
#include <hpx/actions_base/traits/is_valid_action.hpp>
#include <hpx/components_base/traits/action_decorate_function.hpp>
#include <hpx/components_base/traits/component_config_data.hpp>
#include <hpx/components_base/traits/component_heap_type.hpp>
#include <hpx/components_base/traits/component_pin_support.hpp>
#include <hpx/components_base/traits/component_supports_migration.hpp>
#include <hpx/components_base/traits/component_type_database.hpp>
#include <hpx/components_base/traits/component_type_is_compatible.hpp>
#include <hpx/components_base/traits/is_component.hpp>
#include <hpx/functional/traits/is_action.hpp>
#include <hpx/futures/traits/get_remote_result.hpp>
#include <hpx/futures/traits/promise_remote_result.hpp>
#include <hpx/traits/action_message_handler.hpp>
#include <hpx/traits/action_serialization_filter.hpp>
#include <hpx/traits/is_client.hpp>
#include <hpx/traits/is_distribution_policy.hpp>
#include <hpx/traits/managed_component_policies.hpp>
#endif
