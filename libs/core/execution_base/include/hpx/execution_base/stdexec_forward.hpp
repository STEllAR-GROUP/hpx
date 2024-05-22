//  Copyright (c) 2023 Isidoros Tsaousis-Seiras
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#if defined(HPX_HAVE_STDEXEC)
#include <stdexec/execution.hpp>

namespace hpx::execution::experimental {
    // Domain
    using stdexec::default_domain;

    // Receiver
    using stdexec::set_value_t;
    using stdexec::set_error_t;
    using stdexec::set_stopped_t;

    using stdexec::set_value;
    using stdexec::set_error;
    using stdexec::set_stopped;

    using stdexec::receiver_t;
    using stdexec::enable_receiver;

    // Environment
    using stdexec::get_env_t;
    using stdexec::get_env;

    using stdexec::empty_env;
    using stdexec::env_of_t;

//    template <class EnvProvider>
//    using env_of_t = stdexec::env_of_t<EnvProvider>;

    // Queries
    using stdexec::forward_progress_guarantee;

    using stdexec::forwarding_query_t;
    using stdexec::execute_may_block_caller_t;
    using stdexec::get_forward_progress_guarantee_t;
    using stdexec::get_allocator_t;
    using stdexec::get_scheduler_t;
    using stdexec::get_delegatee_scheduler_t;
    using stdexec::get_stop_token_t;
    using stdexec::get_completion_scheduler_t;

    using stdexec::forwarding_query;
    using stdexec::execute_may_block_caller;
    using stdexec::get_forward_progress_guarantee;
    using stdexec::get_allocator;
    using stdexec::get_scheduler;
    using stdexec::get_delegatee_scheduler;
    using stdexec::get_stop_token;
    using stdexec::get_completion_scheduler;

    using stdexec::never_stop_token;
    using stdexec::inplace_stop_source;
    using stdexec::inplace_stop_token;
    using stdexec::in_place_stop_callback;

    using stdexec::stop_token_of_t;

    using stdexec::completion_signatures;
    using stdexec::get_completion_signatures_t;

    // Sender
    using stdexec::connect_t;
    using stdexec::connect;
    using stdexec::connect_result_t;

    using stdexec::sender_t;
    using stdexec::enable_sender;

    // Start
    using stdexec::start_t;
    using stdexec::start;

    // Schedule
    using stdexec::schedule_t;
    using stdexec::schedule;

    using stdexec::schedule_result_t;

    // As awaitable
    using stdexec::as_awaitable_t;
    using stdexec::as_awaitable;

    // Start on
    using stdexec::start_on_t;
    using stdexec::start_on;

    using stdexec::on_t;
    using stdexec::on;

    // Continue on
    using stdexec::continue_on_t;
    using stdexec::continue_on;

    // Transfer just
    using stdexec::transfer_just_t;
    using stdexec::transfer_just;

    // Bulk
//    using stdexec::bulk_t;
//    using stdexec::bulk;

    // Split
    using stdexec::split_t;
    using stdexec::split;

    // Ensure started
    using stdexec::ensure_started_t;
    using stdexec::ensure_started;

    // Transfer
    using stdexec::transfer_t;
    using stdexec::transfer;

    // Tags
    namespace tags {
        using namespace stdexec::tags;
    }

    // Domain
    using stdexec::default_domain;
    using stdexec::dependent_domain;

    // Execute
    using stdexec::execute_t;
    using stdexec::execute;

     // Into Variant
    using stdexec::into_variant_t;
    using stdexec::into_variant;

    // Just
    using stdexec::just_t;
    using stdexec::just_error_t;
    using stdexec::just_stopped_t;

    using stdexec::just;
    using stdexec::just_error;
    using stdexec::just_stopped;

    // Let
    using stdexec::let_value_t;
    using stdexec::let_error_t;
    using stdexec::let_stopped_t;

    using stdexec::let_value;
    using stdexec::let_error;
    using stdexec::let_stopped;

    // Run loop
    using stdexec::run_loop;

    // Schedule from
    using stdexec::schedule_from_t;
    using stdexec::schedule_from;

    // Start detached
    using stdexec::start_detached_t;
    using stdexec::start_detached;

    // Stop token
    using stdexec::stop_callback_for_t;
    using stdexec::stoppable_token;
    using stdexec::stoppable_token_for;
    using stdexec::unstoppable_token;

    // Stopped as error
    using stdexec::stopped_as_error_t;
    using stdexec::stopped_as_error;

    // Stopped as optional
    using stdexec::stopped_as_optional_t;
    using stdexec::stopped_as_optional;

    // Sync wait
    using stdexec::sync_wait_t;
    using stdexec::sync_wait;

    // Sync wait with variant
    using stdexec::sync_wait_with_variant_t;
    using stdexec::sync_wait_with_variant;

    // Then
    using stdexec::then_t;
    using stdexec::then;

    // Transfer just
    using stdexec::transfer_just_t;
    using stdexec::transfer_just;

    // Completion signature manipulators
    using stdexec::completion_signatures_of_t;
    using stdexec::value_types_of_t;
    using stdexec::error_types_of_t;
    using stdexec::sends_stopped;

    using stdexec::make_completion_signatures;
    using stdexec::transform_completion_signatures_of;

    // Transform sender
    using stdexec::transform_sender_t;
    using stdexec::transform_sender;
    using stdexec::transform_sender_result_t;
    using stdexec::transform_env;

    using stdexec::apply_sender_t;
    using stdexec::apply_sender;
    using stdexec::apply_sender_result_t;

    // Upon error
    using stdexec::upon_error_t;
    using stdexec::upon_error;

    // Upon stopped
    using stdexec::upon_stopped_t;
    using stdexec::upon_stopped;

    // When all
    using stdexec::when_all_t;
    using stdexec::when_all;

    using stdexec::when_all_with_variant_t;
    using stdexec::when_all_with_variant;

    using stdexec::transfer_when_all_t;
    using stdexec::transfer_when_all;

    using stdexec::transfer_when_all_with_variant_t;
    using stdexec::transfer_when_all_with_variant;

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

    namespace stdexec_internal {
        using stdexec::__single_sender_value_t;

        namespace __connect_awaitable_ {
            using namespace stdexec::__connect_awaitable_;
        }

        using stdexec::__connect_awaitable_t;
    }
}

namespace hpx::this_thread {

}
#endif