//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// TODO(daissgr) Add file doc 
//
#pragma once

#include <hpx/config.hpp>
#include <hpx/functional/move_only_function.hpp>
#include <hpx/threading_base/thread_pool_base.hpp>

#include <string>

// TODO(daissgr) Really include this here? This will spill into user code (but is sort of required?)
#include <CL/sycl.hpp> 

namespace hpx { namespace sycl { namespace experimental { namespace detail {

    /// Type of the event_callback function used. Unlike the CUDA counterpart we are
    /// not using an error code parameter as SYCL does not provide one
    using event_callback_function_type =
        hpx::move_only_function<void(void)>;

    /// Add callback to be called when all commands currently in the queue are done
    /** Adds an event to the queue by submitting a SYCL dummy kernel and using
     * its return value event. This event will be completely once the dummy kernel
     * and thus all earlier commands/kernels in the queue have fnished.
     * This event is added to the sycl event_callback queue/vector.
     * Thus, the event is polled periodically by the scheduler.
     * When done, the callback function f will be called.
     * Intended to be used with a callback function that sets the future data of an
     * hpx::future representing the completion of an asynchronous SYCL kernel call.
    */
    HPX_CORE_EXPORT void add_event_callback(
        event_callback_function_type&& f, cl::sycl::queue command_queue);

    /// Add callback to be called when all commands up to and including
    /// the passed event are done
    /** Adds an SYCL event directly to the sycl event_callback queue/vector
     * (without using a dummy kernel).
     * The event is polled periodically by the scheduler.
     * When done, the callback function f will be called.
     * Intended to be used with a callback function that sets the future data of an
     * hpx::future representing the completion of an asynchronous SYCL kernel call.
     * NOTE: For hipsycl it is required to flush the internal DAG of the queue, the event
     * thus should be assiociated with the passed queue
    */
    HPX_CORE_EXPORT void add_event_callback(event_callback_function_type&& f,
        cl::sycl::queue command_queue, cl::sycl::event event);

    /// Register SYCL event polling function with the scheduler (see scheduler_base.hpp)
    HPX_CORE_EXPORT void register_polling(hpx::threads::thread_pool_base& pool);
    /// Unregister SYCL event polling function -- only use when all kernels are done
    HPX_CORE_EXPORT void unregister_polling(
        hpx::threads::thread_pool_base& pool);
}}}}    // namespace hpx::sycl::experimental::detail
