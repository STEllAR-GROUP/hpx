//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// TODO(daissgr) Add file doc 
//
#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/concurrency/concurrentqueue.hpp>
#include <hpx/synchronization/spinlock.hpp>
#include <hpx/threading_base/scheduler_base.hpp>
#include <hpx/threading_base/thread_pool_base.hpp>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <hpx/async_sycl/detail/sycl_event_callback.hpp>
#if defined(__HIPSYCL__) 
// Lots of warning within the hipsycl headers
// To compiler with Werror we need to disable those
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-copy"
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wdouble-promotion"
#pragma clang diagnostic ignored "-Wunused-local-typedef"
#pragma clang diagnostic ignored "-Wsign-compare"
#pragma clang diagnostic ignored "-Wgcc-compat"
#pragma clang diagnostic ignored "-Wembedded-directive"
#pragma clang diagnostic ignored "-Wmismatched-tags"
#pragma clang diagnostic ignored "-Wreorder-ctor"
#endif
#ifndef __SYCL_DEVICE_ONLY__

namespace hpx { namespace sycl { namespace experimental { namespace detail {

    // this code runs on a std::thread, but we will use a spinlock
    // as we never suspend - only ever try_lock, or exit
    using mutex_type = hpx::spinlock;
    mutex_type& get_vector_mtx()
    {
        static mutex_type vector_mtx;
        return vector_mtx;
    }

#if defined(HPX_DEBUG)
    std::atomic<std::size_t>& get_register_polling_count()
    {
        static std::atomic<std::size_t> register_polling_count{0};
        return register_polling_count;
    }
#endif

    // -------------------------------------------------------------
    /// Holds a SYCL event and a callback. The callback is intended to be called when
    /// the event is ready.
    struct event_callback
    {
        cl::sycl::event event;
        event_callback_function_type f;
    };

    // -------------------------------------------------------------
    /// Event_callbacks are added using a concurrent queue
    using event_callback_queue_type =
        concurrency::ConcurrentQueue<event_callback>;
    /// Event_callbacks stored in a callback vector for later checking 
    using event_callback_vector_type = std::vector<event_callback>;
    /// Access to (static) event callback vector containing unfinished events
    event_callback_vector_type& get_event_callback_vector()
    {
        static event_callback_vector_type event_callback_vector;
        return event_callback_vector;
    }

    /// Access to (static) event callback queue containing unfinished events
    event_callback_queue_type& get_event_callback_queue()
    {
        static event_callback_queue_type event_callback_queue;
        return event_callback_queue;
    }

    /// Get the rough number of events (required by get_work_count() functionality)
    std::size_t get_number_of_enqueued_events()
    {
        return get_event_callback_queue().size_approx();
    }

    /// Get the number of events (required by get_work_count() functionality)
    std::size_t get_number_of_active_events()
    {
        return get_event_callback_vector().size();
    }
    /// Add event_callback into queue for checking
    void add_to_event_callback_queue(event_callback&& continuation)
    {
        HPX_ASSERT_MSG(get_register_polling_count() != 0,
            "SYCL event polling has not been enabled on any pool. Make sure "
            "that SYCL event polling is enabled on at least one thread pool.");

        get_event_callback_queue().enqueue(HPX_MOVE(continuation));

    }
    // In case event_callback was not done when exiting the queue
    // it will be added to the vector using this method for later checking
    void add_to_event_callback_vector(event_callback&& continuation)
    {
        get_event_callback_vector().push_back(HPX_MOVE(continuation));

    }

    // -------------------------------------------------------------
    // Functions to add event_callbacks to the event_callback queue
    // Either using a dummy kernel within the SYCL command queue (to get a sycl event)
    // or by supplying a SYCL event directly

    // Adds an event callback directly for a given event 
    // (event needs to be from the queue in question when using hipsycl)
    void add_event_callback(
        event_callback_function_type&& f, cl::sycl::queue command_queue,
        cl::sycl::event event)
    {
        detail::add_to_event_callback_queue(event_callback{event, HPX_MOVE(f)});

#if defined(__HIPSYCL__) 
        // See https://github.com/illuhad/hipSYCL/issues/599 for why we need to flush
        // See https://github.com/illuhad/hipSYCL/pull/749 for API change: 
        //
        // The latter PR also dictates our minimal required hipsycl version.
        //
        // We COULD support older hipsycl version by using an API switch depending on the
        // hipsycl version but since we do not need to support any old SYCL code I do not
        // see a reason for doing so.
        command_queue.get_context().hipSYCL_runtime()->dag().flush_async();
#endif
    }

    // -------------------------------------------------------------
    // Function to be called from the scheduler: 
    // poll (checks if any event in the vector/queue is done and calls its callback
    // get_work_count (how many unfinished events are left (approx)
    
    /// Check for completed SYCL events and call their associated callbacks.
    /** This methods trys to get the lock for the callback vector. 
     * If unsuccessful it will exit as another thread is already polling.
     * If successful it will first check the event_callback_vector for
     * any events that are completed/finished and call their respective callbacks.
     * Afterwards the queue is checked: If an event here is done, the callback
     * will also be invoked. Otherwise the assiocated event_callback is added to
     * the callback vector to be checked a later time.
     *
     * Unlike the CUDA counterpart, no event pool is used here since we have to work
     * with the SYCL events directly returned form the SYCL runtime
     * (as there is no syclEventRecord). We have to trust in the respective SYCL 
     * implementation to use an event pool internally.
     */
    hpx::threads::policies::detail::polling_status poll()
    {
        using hpx::threads::policies::detail::polling_status;

        auto& event_callback_vector = detail::get_event_callback_vector();

        // Don't poll if another thread is already polling
        std::unique_lock<hpx::sycl::experimental::detail::mutex_type> lk(
            detail::get_vector_mtx(), std::try_to_lock);
        if (!lk.owns_lock())
        {
            return polling_status::idle;
        }

        // Iterate over our list of events and see if any have completed
        event_callback_vector.erase(
            std::remove_if(event_callback_vector.begin(),
                event_callback_vector.end(),
                [&](event_callback& continuation) {
                    const auto event_status = continuation.event.get_info<
                        cl::sycl::info::event::command_execution_status>();

                    if (event_status != cl::sycl::info::event_command_status::complete)
                    {
                        return false;
                    }
                    continuation.f();
                    return true;
                }),
            event_callback_vector.end());

        // Check queue for completed events next
        // If not completed: Add to the event_callback vector for later checking
        detail::event_callback continuation;
        while (detail::get_event_callback_queue().try_dequeue(continuation))
        {
            const auto event_status = continuation.event.get_info<
               cl::sycl::info::event::command_execution_status>();

            if (event_status != cl::sycl::info::event_command_status::complete)
            {
                add_to_event_callback_vector(HPX_MOVE(continuation));
            }
            else
            {
                continuation.f();
            }
        }

        using hpx::threads::policies::detail::polling_status;
        return get_event_callback_vector().empty() ? polling_status::idle :
                                                     polling_status::busy;
    }

    /// Gets the number of events left in the vector (if not locked by another thread) in
    /// addition the the approximated number of events in the queue
    std::size_t get_work_count()
    {
        std::size_t work_count = 0;
        {
            std::unique_lock<mutex_type> lk(get_vector_mtx(), std::try_to_lock);
            if (lk.owns_lock())
            {
                work_count += get_number_of_active_events();
            }
        }

        work_count += get_number_of_enqueued_events();

        return work_count;
    }

    // -------------------------------------------------------------
    /// Register SYCL event polling function with the scheduler (see scheduler_base.hpp)
    void register_polling(hpx::threads::thread_pool_base& pool)
    {
#if defined(HPX_DEBUG)
        ++get_register_polling_count();
#endif
        auto* sched = pool.get_scheduler();
        sched->set_sycl_polling_functions(
            &hpx::sycl::experimental::detail::poll, &get_work_count);
    }

    // -------------------------------------------------------------
    /// Unregister SYCL event polling function -- only use when all kernels are done
    void unregister_polling(hpx::threads::thread_pool_base& pool)
    {
#if defined(HPX_DEBUG)
        {
            std::unique_lock<hpx::sycl::experimental::detail::mutex_type> lk(
                detail::get_vector_mtx());
            bool event_queue_empty =
                get_event_callback_queue().size_approx() == 0;
            bool event_vector_empty = get_event_callback_vector().empty();
            lk.unlock();
            HPX_ASSERT_MSG(event_queue_empty,
                "SYCL event polling was disabled while there are unprocessed "
                "SYCL events. Make sure SYCL event polling is not disabled too "
                "early.");
            HPX_ASSERT_MSG(event_vector_empty,
                "SYCL event polling was disabled while there are unprocessed "
                "SYCL events. Make sure SYCL event polling is not disabled too "
                "early.");
        }
#endif
        auto* sched = pool.get_scheduler();
        sched->clear_sycl_polling_function();
    }
}}}}    // namespace hpx::sycl::experimental::detail
#endif
