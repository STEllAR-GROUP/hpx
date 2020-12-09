//  Copyright (c) 2020 John Biddiscombe
//  Copyright (c) 2016 Hartmut Kaiser
//  Copyright (c) 2016 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#include <hpx/allocator_support/internal_allocator.hpp>
#include <hpx/assert.hpp>
#include <hpx/async_cuda/cuda_event.hpp>
#include <hpx/async_cuda/target.hpp>
#include <hpx/futures/traits/future_access.hpp>
#include <hpx/modules/errors.hpp>
#if defined(HPX_HAVE_DISTRIBUTED_RUNTIME)
#include <hpx/naming_base/id_type.hpp>
#include <hpx/runtime/find_here.hpp>
#endif
#include <hpx/runtime_fwd.hpp>
#include <hpx/threading_base/thread_helpers.hpp>

#include <atomic>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <hpx/async_cuda/custom_gpu_api.hpp>

namespace hpx { namespace cuda { namespace experimental { namespace detail {

    mutex_type& get_list_mtx()
    {
        static mutex_type list_mtx;
        return list_mtx;
    }

    runtime_registration_wrapper::runtime_registration_wrapper(hpx::runtime* rt)
      : rt_(rt)
      , registered_(false)
    {
        if (nullptr != hpx::get_runtime_ptr())
        {
            return;
        }

        HPX_ASSERT(rt);

        // Register this thread with HPX, this should be done once for
        // each external OS-thread intended to invoke HPX functionality.
        // Calling this function more than once on the same thread will
        // report an error.
        hpx::error_code ec(hpx::lightweight);    // ignore errors
        hpx::register_thread(rt_, "cuda", ec);
        registered_ = true;
    }

    runtime_registration_wrapper::~runtime_registration_wrapper()
    {
        // Unregister the thread from HPX, this should be done once in the end
        // before the external thread exits, if the runtime registration
        // wrapper actually registered the thread (it may not do so if the
        // wrapper is constructed on a HPX worker thread).
        if (registered_)
        {
            hpx::unregister_thread(rt_);
        }
    }

    // -------------------------------------------------------------
    // main API call to get a future from a stream
    hpx::future<void> get_future_with_callback(cudaStream_t stream)
    {
        return get_future_with_callback(
            hpx::util::internal_allocator<>{}, stream);
    }

#if defined(HPX_DEBUG)
    std::atomic<std::size_t>& get_register_polling_count()
    {
        static std::atomic<std::size_t> register_polling_count{0};
        return register_polling_count;
    }
#endif

    // -------------------------------------------------------------
    // main API call to get a future from a stream
    hpx::future<void> get_future_with_event(cudaStream_t stream)
    {
        HPX_ASSERT_MSG(get_register_polling_count() != 0,
            "CUDA event polling has not been enabled on any pool. Make sure "
            "that CUDA event polling is enabled on at least one thread pool.");
        return get_future_with_event(hpx::util::internal_allocator<>{}, stream);
    }

    // -------------------------------------------------------------
    std::vector<future_data_ptr>& get_active_futures()
    {
        static std::vector<future_data_ptr> active_futures;
        return active_futures;
    }

    // -------------------------------------------------------------
    queue_type& get_event_queue()
    {
        static queue_type event_queue;
        return event_queue;
    }

    // -------------------------------------------------------------
    // used internally to add a cuda event to the lockfree queue
    // that will be used by the polling routines to check when requests
    // have completed
    void add_to_event_queue(future_data_ptr data)
    {
        // place this future data request in our queue for handling
        get_event_queue().enqueue(data);

        // clang-format off
        cud_debug.debug(debug::str<>("event queued")
            , "request", debug::hex<8>(data->event_)
            , "futures", debug::dec<3>(get_active_futures().size()));
        // clang-format on
    }

    // -------------------------------------------------------------
    // used internally to add a request to the main polling vector/list
    // that is used to call cudaEventQuery
    void add_to_polling_list(future_data_ptr data)
    {
        // this will make a copy and increment the ref count
        get_active_futures().push_back(data);

        // clang-format off
        cud_debug.debug(debug::str<>("event -> poll")
            , "event", debug::hex<8>(data->event_)
            , "futures", debug::dec<3>(get_active_futures().size()));
        // clang-format on
    }

    // -------------------------------------------------------------
    // Background progress function for async CUDA operations
    // Checks for completed cudaEvent_t and sets cuda::future
    // ready when found.
    // We process outstanding futures in the polling list first,
    // then any new future requests are polled and if not ready
    // added to the polling list (for next time)
    void poll()
    {
        // don't poll if another thread is already polling
        std::unique_lock<hpx::cuda::experimental::detail::mutex_type> lk(
            detail::get_list_mtx(), std::try_to_lock);
        if (!lk.owns_lock())
        {
            if (cud_debug.is_enabled())
            {
                // for debugging
                static auto poll_deb =
                    cud_debug.make_timer(1, debug::str<>("Poll - lock failed"));
                // clang-format off
                cud_debug.timed(poll_deb,
                    "futures", debug::dec<3>(get_active_futures().size()));
                // clang-format on
            }
            return;
        }

        auto& future_vec = detail::get_active_futures();

        if (cud_debug.is_enabled())
        {
            // for debugging
            static auto poll_deb =
                cud_debug.make_timer(1, debug::str<>("Poll - lock success"));
            // clang-format off
             cud_debug.timed(poll_deb,
                 "futures", debug::dec<3>(get_active_futures().size()));
            // clang-format on
        }

        // grab the handle to the event pool so we can return completed events
        cuda_event_pool& pool =
            hpx::cuda::experimental::cuda_event_pool::get_event_pool();

        // iterate over our list of events and see if any have completed
        detail::future_data_ptr fdp;
        using i_type = std::vector<future_data_ptr>::iterator;
        for (i_type it = future_vec.begin(); it != future_vec.end();)
        {
            fdp = *it;
            cudaError_t status = cudaEventQuery(fdp->event_);
            if (status == cudaErrorNotReady)
            {
                // this event has not been triggered yet
                continue;
            }
            else if (status == cudaSuccess)
            {
                fdp->set_data(hpx::util::unused);
                // clang-format off
                cud_debug.debug(debug::str<>("set ready vector")
                    , "event", debug::hex<8>(fdp->event_)
                    , "futures", debug::dec<3>(get_active_futures().size()));
                // clang-format on
                // drop future and reuse event
                it = future_vec.erase(it);
                pool.push(fdp->event_);
            }
            else
            {
                // clang-format off
                cud_debug.debug(debug::str<>("set exception vector")
                    , "event", debug::hex<8>(fdp->event_)
                    , "futures", debug::dec<3>(get_active_futures().size()));
                // clang-format on
                fdp->set_exception(std::make_exception_ptr(cuda_exception(
                    std::string("cuda function returned error code :") +
                        cudaGetErrorString(status),
                    status)));
                ++it;
            }
        }

        // have any requests been made that need to be handled?
        // if so, move them all from the lockfree request list, onto the
        // polling list
        while (detail::get_event_queue().try_dequeue(fdp))
        {
            cudaError_t status = cudaEventQuery(fdp->event_);
            if (status == cudaErrorNotReady)
            {
                // this event has not been triggered yet
                add_to_polling_list(std::move(fdp));
                continue;
            }
            else if (status == cudaSuccess)
            {
                fdp->set_data(hpx::util::unused);
                // clang-format off
                cud_debug.debug(debug::str<>("set ready queue")
                    , "event", debug::hex<8>(fdp->event_)
                    , "futures", debug::dec<3>(get_active_futures().size()));
                // clang-format on
                // event is done and can be reused
                pool.push(fdp->event_);
            }
            else
            {
                // clang-format off
                cud_debug.debug(debug::str<>("set exception queue")
                    , "event", debug::hex<8>(fdp->event_)
                    , "futures", debug::dec<3>(get_active_futures().size()));
                // clang-format on
                fdp->set_exception(std::make_exception_ptr(cuda_exception(
                    std::string("cuda function returned error code :") +
                        cudaGetErrorString(status),
                    status)));
            }
        }
    }

    // -------------------------------------------------------------
    void register_polling(hpx::threads::thread_pool_base& pool)
    {
#if defined(HPX_DEBUG)
        ++get_register_polling_count();
#endif
        cud_debug.debug(debug::str<>("enable polling"));
        auto* sched = pool.get_scheduler();
        sched->set_cuda_polling_function(
            &hpx::cuda::experimental::detail::poll);
    }

    // -------------------------------------------------------------
    void unregister_polling(hpx::threads::thread_pool_base& pool)
    {
#if defined(HPX_DEBUG)
        {
            std::unique_lock<hpx::cuda::experimental::detail::mutex_type> lk(
                detail::get_list_mtx());
            bool event_queue_empty = get_event_queue().size_approx() == 0;
            bool active_futures_empty = get_active_futures().empty();
            lk.unlock();
            HPX_ASSERT_MSG(event_queue_empty,
                "CUDA event polling was disabled while there are unprocessed "
                "CUDA events. Make sure CUDA event polling is not disabled too "
                "early.");
            HPX_ASSERT_MSG(active_futures_empty,
                "CUDA event polling was disabled while there are active CUDA "
                "futures. Make sure CUDA event polling is not disabled too "
                "early.");
        }
#endif
        cud_debug.debug(debug::str<>("disable polling"));
        auto* sched = pool.get_scheduler();
        sched->clear_cuda_polling_function();
    }
}}}}    // namespace hpx::cuda::experimental::detail
