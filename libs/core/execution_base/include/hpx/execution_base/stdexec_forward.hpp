//  Copyright (c) 2024 Isidoros Tsaousis-Seiras
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#include <hpx/config.hpp>

#if defined(HPX_HAVE_STDEXEC)

/* TODO: Find out what diagnostics should be disabled for stdexec to compile.
 * currently it seems to need at least "-Wgnu-zero-variadic-macro-arguments"
 * and "-Wmissing-braces" even though they are explicitly disabled inside
 * stdexec.
 *
 * Clang needs "-Wgnu-zero-variadic-macro-arguments", but this option does not
 * exist in GCC. Will leave this commented out for now.
 *
 * #if defined(HPX_CLANG_VERSION)
 * #pragma GCC diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
 * #endif
 */

#if defined(HPX_GCC_VERSION)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wextra"
#pragma GCC diagnostic ignored "-Wmissing-braces"
#elif defined(HPX_CLANG_VERSION)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#endif

#include <stdexec/execution.hpp>

#if defined(HPX_GCC_VERSION)
#pragma GCC diagnostic pop
#elif defined(HPX_CLANG_VERSION)
#pragma clang diagnostic pop
#endif

namespace hpx::execution::experimental {
    // Domain
    using stdexec::default_domain;

    // Receiver
    using stdexec::set_error_t;
    using stdexec::set_stopped_t;
    using stdexec::set_value_t;

    using stdexec::set_error;
    using stdexec::set_stopped;
    using stdexec::set_value;

    using stdexec::enable_receiver;
    using stdexec::receiver_t;

    // Environment
    using stdexec::get_env;
    using stdexec::get_env_t;

    using stdexec::empty_env;
    using stdexec::env_of_t;

    using stdexec::env;
    using stdexec::prop;

    // Queries
    using stdexec::forward_progress_guarantee;

    using stdexec::execute_may_block_caller_t;
    using stdexec::forwarding_query_t;
    using stdexec::get_allocator_t;
    using stdexec::get_completion_scheduler_t;
    using stdexec::get_delegatee_scheduler_t;
    using stdexec::get_forward_progress_guarantee_t;
    using stdexec::get_scheduler_t;
    using stdexec::get_stop_token_t;

    using stdexec::execute_may_block_caller;
    using stdexec::forwarding_query;
    using stdexec::get_allocator;
    using stdexec::get_completion_scheduler;
    using stdexec::get_delegatee_scheduler;
    using stdexec::get_forward_progress_guarantee;
    using stdexec::get_scheduler;
    using stdexec::get_stop_token;

    using stdexec::in_place_stop_callback;
    using stdexec::inplace_stop_source;
    using stdexec::inplace_stop_token;
    using stdexec::never_stop_token;

    using stdexec::stop_token_of_t;

    using stdexec::completion_signatures;
    using stdexec::get_completion_signatures;
    using stdexec::get_completion_signatures_t;

    // Operation State
    using stdexec::operation_state_t;

    // Sender
    using stdexec::connect;
    using stdexec::connect_result_t;
    using stdexec::connect_t;

    using stdexec::enable_sender;
    using stdexec::sender_t;

    // Start
    using stdexec::start;
    using stdexec::start_t;

    // Schedule
    using stdexec::schedule;
    using stdexec::schedule_t;

    using stdexec::schedule_result_t;

    // As awaitable
    using stdexec::as_awaitable;
    using stdexec::as_awaitable_t;

    // Start on
    using stdexec::start_on;
    using stdexec::start_on_t;

    using stdexec::on;
    using stdexec::on_t;

    // Continue on
    using stdexec::continue_on;
    using stdexec::continue_on_t;

    // Transfer just
    using stdexec::transfer_just;
    using stdexec::transfer_just_t;

    // Bulk (NOT FORWARDED)
    //    using stdexec::bulk_t;
    //    using stdexec::bulk;

    // Split
    using stdexec::split;
    using stdexec::split_t;

    // Ensure started
    using stdexec::ensure_started;
    using stdexec::ensure_started_t;

    // Transfer
    using stdexec::transfer;
    using stdexec::transfer_t;

    // Tags
    namespace tags {
        using namespace stdexec::tags;
    }

    // Domain
    using stdexec::default_domain;
    using stdexec::dependent_domain;

    // Execute
    using stdexec::execute;
    using stdexec::execute_t;

    // Into Variant
    using stdexec::into_variant;
    using stdexec::into_variant_t;

    // Just
    using stdexec::just_error_t;
    using stdexec::just_stopped_t;
    using stdexec::just_t;

    using stdexec::just;
    using stdexec::just_error;
    using stdexec::just_stopped;

    // Let
    using stdexec::let_error_t;
    using stdexec::let_stopped_t;
    using stdexec::let_value_t;

    using stdexec::let_error;
    using stdexec::let_stopped;
    using stdexec::let_value;

    // Run loop
    using stdexec::run_loop;

    // Schedule from
    using stdexec::schedule_from;
    using stdexec::schedule_from_t;

    // Start detached
    using stdexec::start_detached;
    using stdexec::start_detached_t;

    // Stop token
    using stdexec::stop_callback_for_t;
    using stdexec::stoppable_token;
    using stdexec::stoppable_token_for;
    using stdexec::unstoppable_token;

    // Stopped as error
    using stdexec::stopped_as_error;
    using stdexec::stopped_as_error_t;

    // Stopped as optional
    using stdexec::stopped_as_optional;
    using stdexec::stopped_as_optional_t;

    // Sync wait
    using stdexec::sync_wait;
    using stdexec::sync_wait_t;

    // Sync wait with variant
    using stdexec::sync_wait_with_variant;
    using stdexec::sync_wait_with_variant_t;

    // Then
    using stdexec::then;
    using stdexec::then_t;

    // Transfer just
    using stdexec::transfer_just;
    using stdexec::transfer_just_t;

    // Completion signature manipulators
    using stdexec::completion_signatures_of_t;
    using stdexec::error_types_of_t;
    using stdexec::sends_stopped;
    using stdexec::value_types_of_t;

    using stdexec::make_completion_signatures;
    using stdexec::transform_completion_signatures;
    using stdexec::transform_completion_signatures_of;

    // Transform sender
    using stdexec::transform_env;
    using stdexec::transform_sender;
    using stdexec::transform_sender_result_t;
    using stdexec::transform_sender_t;

    using stdexec::apply_sender;
    using stdexec::apply_sender_result_t;
    using stdexec::apply_sender_t;

    // Upon error
    using stdexec::upon_error;
    using stdexec::upon_error_t;

    // Upon stopped
    using stdexec::upon_stopped;
    using stdexec::upon_stopped_t;

    // When all
    using stdexec::when_all;
    using stdexec::when_all_t;

    using stdexec::when_all_with_variant;
    using stdexec::when_all_with_variant_t;

    using stdexec::transfer_when_all;
    using stdexec::transfer_when_all_t;

    using stdexec::transfer_when_all_with_variant;
    using stdexec::transfer_when_all_with_variant_t;

    // With awaitable senders
    using stdexec::with_awaitable_senders;

    // Concepts
    using stdexec::sender;
    using stdexec::sender_in;
    using stdexec::sender_of;
    using stdexec::sender_to;

    using stdexec::receiver;
    using stdexec::receiver_of;

    using stdexec::scheduler;

    using stdexec::operation_state;

    namespace stdexec_non_standard_tag_invoke {
        // Presently, the stdexec repository implements tag invoke,
        // however it includes a non-standard (in the sense of unexpected) extension.
        // tag invoke first checks for the existence of a .query member function or
        // a ::query static function.
        using stdexec::tag_invoke;
        using stdexec::tag_invoke_result;

        using stdexec::nothrow_tag_invocable;
        using stdexec::tag_invocable;
    }    // namespace stdexec_non_standard_tag_invoke

    namespace stdexec_internal {
        using stdexec::__single_sender_value_t;

        namespace __connect_awaitable_ {
            using namespace stdexec::__connect_awaitable_;
        }

        using stdexec::__connect_awaitable_t;
    }    // namespace stdexec_internal
}    // namespace hpx::execution::experimental

// Leaving this as a placeholder
namespace hpx::this_thread {
}
#endif
