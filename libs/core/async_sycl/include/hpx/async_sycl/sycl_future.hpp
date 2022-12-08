//  Copyright (c) 2022 Gregor Daiß
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//
// This file is very similar to its CUDA counterpart (cuda_future.hpp) just
// adapted/simplified) for sycl (we have to get our events from the sycl
// runtime, and normal stream callbacks are not possible with SYCL -- we only
// have the option to do event polling)
//
// hpxinspect:noascii

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/async_sycl/detail/sycl_event_callback.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/modules/concurrency.hpp>
#include <hpx/modules/execution_base.hpp>
#include <hpx/modules/memory.hpp>
#include <hpx/modules/threading_base.hpp>
#include <hpx/runtime_local/runtime_local_fwd.hpp>
#include <hpx/runtime_local/thread_pool_helpers.hpp>

#include <memory>
#include <string>
#include <type_traits>
#include <utility>

namespace hpx { namespace sycl { namespace experimental {
    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        // -------------------------------------------------------------
        // SYCL future data implementation
        // Using an event based callback that must be polled/queried by
        // the runtime to set the future ready state

        template <typename Allocator>
        struct future_data
          : lcos::detail::future_data_allocator<void, Allocator>
        {
            HPX_NON_COPYABLE(future_data);

            using init_no_addref =
                typename lcos::detail::future_data_allocator<void,
                    Allocator>::init_no_addref;

            using other_allocator = typename std::allocator_traits<
                Allocator>::template rebind_alloc<future_data>;

            future_data() = default;

            future_data(init_no_addref no_addref, other_allocator const& alloc,
                cl::sycl::event command_event)
              : lcos::detail::future_data_allocator<void, Allocator>(
                    no_addref, alloc)
            {
                add_event_callback(
                    [fdp = hpx::intrusive_ptr<future_data>(this)]() {
                        fdp->set_data(hpx::util::unused);
                        // TODO Future work considerations: exception handling
                        // in here?  Technically SYCL has asynchronous error
                        // handling (exceptions...) in kernel code but only if
                        // it is running on host code (which we are not
                        // interested in as of now)
                    },
                    command_event);
            }
        };

        // -------------------------------------------------------------
        template <typename Allocator>
        hpx::future<void> get_future(
            Allocator const& a, cl::sycl::event command_event)
        {
            using shared_state = future_data<Allocator>;

            using other_allocator = typename std::allocator_traits<
                Allocator>::template rebind_alloc<shared_state>;
            using traits = std::allocator_traits<other_allocator>;

            using init_no_addref = typename shared_state::init_no_addref;

            using unique_ptr = std::unique_ptr<shared_state,
                util::allocator_deleter<other_allocator>>;

            other_allocator alloc(a);
            unique_ptr p(traits::allocate(alloc, 1),
                hpx::util::allocator_deleter<other_allocator>{alloc});

            traits::construct(
                alloc, p.get(), init_no_addref{}, alloc, command_event);

            return hpx::traits::future_access<future<void>>::create(
                p.release(), false);
        }
        // -------------------------------------------------------------
        // non allocator version of : get future with an event set
        HPX_CORE_EXPORT hpx::future<void> get_future(
            cl::sycl::event command_event);
        // -------------------------------------------------------------
        /// Convenience wrapper to get future from just a queue
        HPX_FORCEINLINE hpx::future<void> get_future(
            cl::sycl::queue& command_queue)
        {
            // The SYCL standard does not include a eventRecord method Instead
            // we have to submit some dummy function and use the event the
            // launch returns
            cl::sycl::event event = command_queue.submit(
                [](cl::sycl::handler& h) { h.single_task([]() {}); });
            return get_future(event);
        }
    }    // namespace detail
}}}      // namespace hpx::sycl::experimental
