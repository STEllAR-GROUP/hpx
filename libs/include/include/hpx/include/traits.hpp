//  Copyright (c) 2007-2019 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_JAN_31_2015_0130PM)
#define HPX_TRAITS_JAN_31_2015_0130PM

#include <hpx/concepts/concepts.hpp>
#include <hpx/concepts/has_member_xxx.hpp>
#include <hpx/concepts/has_xxx.hpp>
#include <hpx/datastructures/traits/is_tuple_like.hpp>
#include <hpx/datastructures/traits/supports_streaming_with_any.hpp>
#include <hpx/execution/traits/is_executor.hpp>
#include <hpx/execution/traits/is_executor_parameters.hpp>
#include <hpx/execution/traits/is_timed_executor.hpp>
#include <hpx/functional/traits/get_function_address.hpp>
#include <hpx/functional/traits/get_function_annotation.hpp>
#include <hpx/functional/traits/is_action.hpp>
#include <hpx/functional/traits/is_bind_expression.hpp>
#include <hpx/functional/traits/is_callable.hpp>
#include <hpx/functional/traits/is_placeholder.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/iterator_support/traits/is_range.hpp>
#include <hpx/plugin/traits/plugin_config_data.hpp>
#include <hpx/serialization/traits/brace_initializable_traits.hpp>
#include <hpx/serialization/traits/is_bitwise_serializable.hpp>
#include <hpx/serialization/traits/needs_automatic_registration.hpp>
#include <hpx/serialization/traits/polymorphic_traits.hpp>
#include <hpx/serialization/traits/serialization_access_data.hpp>
#include <hpx/traits/acquire_future.hpp>
#include <hpx/traits/acquire_shared_state.hpp>
#include <hpx/traits/action_decorate_continuation.hpp>
#include <hpx/traits/action_decorate_function.hpp>
#include <hpx/traits/action_does_termination_detection.hpp>
#include <hpx/traits/action_is_target_valid.hpp>
#include <hpx/traits/action_message_handler.hpp>
#include <hpx/traits/action_priority.hpp>
#include <hpx/traits/action_remote_result.hpp>
#include <hpx/traits/action_schedule_thread.hpp>
#include <hpx/traits/action_select_direct_execution.hpp>
#include <hpx/traits/action_serialization_filter.hpp>
#include <hpx/traits/action_stacksize.hpp>
#include <hpx/traits/action_was_object_migrated.hpp>
#include <hpx/traits/component_config_data.hpp>
#include <hpx/traits/component_heap_type.hpp>
#include <hpx/traits/component_pin_support.hpp>
#include <hpx/traits/component_supports_migration.hpp>
#include <hpx/traits/component_type_database.hpp>
#include <hpx/traits/component_type_is_compatible.hpp>
#include <hpx/traits/extract_action.hpp>
#include <hpx/traits/future_access.hpp>
#include <hpx/traits/future_traits.hpp>
#include <hpx/traits/get_remote_result.hpp>
#include <hpx/traits/is_client.hpp>
#include <hpx/traits/is_component.hpp>
#include <hpx/traits/is_continuation.hpp>
#include <hpx/traits/is_distribution_policy.hpp>
#include <hpx/traits/is_future.hpp>
#include <hpx/traits/is_future_range.hpp>
#include <hpx/traits/is_future_tuple.hpp>
#include <hpx/traits/is_launch_policy.hpp>
#include <hpx/traits/is_valid_action.hpp>
#include <hpx/traits/managed_component_policies.hpp>
#include <hpx/traits/pointer_category.hpp>
#include <hpx/traits/promise_local_result.hpp>
#include <hpx/traits/promise_remote_result.hpp>
#include <hpx/traits/segmented_iterator_traits.hpp>
#include <hpx/type_support/detail/wrap_int.hpp>

#endif
