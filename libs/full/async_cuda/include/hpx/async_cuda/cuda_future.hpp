//  Copyright (c) 2020 John Biddiscombe
//  Copyright (c) 2016 Thomas Heller
//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/async_cuda/cuda_event.hpp>
#include <hpx/async_cuda/cuda_exception.hpp>
#include <hpx/debugging/print.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/modules/concurrency.hpp>
#include <hpx/modules/execution_base.hpp>
#include <hpx/modules/memory.hpp>
#include <hpx/modules/threading_base.hpp>
#include <hpx/runtime/runtime_fwd.hpp>
#include <hpx/runtime_local/thread_pool_helpers.hpp>
//
#include <cuda_runtime.h>
//
#include <cstddef>
#include <iosfwd>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace cuda { namespace experimental {

    using print_on = debug::enable_print<false>;
    static constexpr print_on cud_debug("CUDAFUT");

    // clang-format off
    using event_mode = std::true_type;
    using callback_mode = std::false_type;
    // clang-format on

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {

        // this code runs on a std::thread, but we will use a spinlock
        // as we never suspend - only ever try_lock, or exit
        using mutex_type = hpx::lcos::local::spinlock;
        using cuda_event_type = cudaEvent_t;

        // mutex needed to protect mpi request list
        HPX_EXPORT mutex_type& get_list_mtx();

        // -------------------------------------------------------------
        // a callback on an NVidia cuda thread should be registered with
        // hpx to ensure any thread local operations are valid
        // @TODO - get rid of this
        struct runtime_registration_wrapper
        {
            runtime_registration_wrapper(hpx::runtime* rt);
            ~runtime_registration_wrapper();
            hpx::runtime* rt_;
        };

        // -------------------------------------------------------------
        template <typename Allocator>
        struct future_data;

        // -------------------------------------------------------------
        // helper struct to delete future data in destructor
        template <typename Allocator>
        struct release_on_exit
        {
            release_on_exit(future_data<Allocator>* data)
              : data_(data)
            {
            }

            ~release_on_exit()
            {
                // release the shared state
                lcos::detail::intrusive_ptr_release(data_);
            }

            future_data<Allocator>* data_;
        };

        // -------------------------------------------------------------
        // cuda future data implementation
        // This version supports 2 modes of operation
        // 1) a callback based future that is made ready
        // by a cuda callback when the stream event occurs
        // 2) an even based callback that must be polled/queried by
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

            future_data()
              : rt_(hpx::get_runtime_ptr())
            {
            }

            // constructor used by callback based futures
            future_data(init_no_addref no_addref, other_allocator const& alloc,
                cudaStream_t stream, callback_mode)
              : lcos::detail::future_data_allocator<void, Allocator>(
                    no_addref, alloc)
              , rt_(hpx::get_runtime_ptr())
            {
                init_callback(stream);
            }

            // constructor used by event based futures
            future_data(init_no_addref no_addref, other_allocator const& alloc,
                cudaStream_t stream, event_mode)
              : lcos::detail::future_data_allocator<void, Allocator>(
                    no_addref, alloc)
            {
                init_event(stream);
            }

            // @TODO - another thread might queue a kernel function
            // at the same time, on the same stream - causing our event
            // to be later than expected :
            // should there be a mutex protecting future creation + event record?
            void init_event(cudaStream_t stream)
            {
                // even if the event is triggered/ready before we complete, we do
                // not need to worry about a race, because we only poll for the complete
                // event after returning from here
                if (!cuda_event_pool::get_event_pool().pop(event_))
                {
                    HPX_THROW_EXCEPTION(invalid_status, "init_event",
                        "cuda event stack size has been exceeded "
                        "recompile with a larger event stack size");
                }
                check_cuda_error(cudaEventRecord(event_, stream));

                cud_debug.debug(
                    debug::str<>("init_event"), "event", debug::hex<8>(event_));
            }

            void init_callback(cudaStream_t stream)
            {
                // Hold on to the shared state on behalf of the cuda runtime
                // right away as the callback could be called immediately
                // and it happens on another thread which might cause a race
                lcos::detail::intrusive_ptr_add_ref(this);

                cudaError_t error =
                    cudaStreamAddCallback(stream, stream_callback, this, 0);
                if (error != cudaSuccess)
                {
                    // callback was not called, release object
                    lcos::detail::intrusive_ptr_release(this);
                    // report error
                    check_cuda_error(error);
                }
                cud_debug.debug(
                    debug::str<>("init_callback"), "event", debug::ptr(this));
            }

            // this is called from the nvidia backend on a non-hpx thread
            // extreme care must be taken to not lock/block or take too long
            // in this callback
            static void CUDART_CB stream_callback(
                cudaStream_t stream, cudaError_t error, void* user_data)
            {
                future_data* this_ = static_cast<future_data*>(user_data);

                runtime_registration_wrapper wrap(this_->rt_);
                release_on_exit<Allocator> on_exit(this_);

                if (error != cudaSuccess)
                {
                    this_->set_exception(HPX_GET_EXCEPTION(kernel_error,
                        "cuda::detail::future_data::stream_callback()",
                        std::string("cudaStreamAddCallback failed: ") +
                            cudaGetErrorString(error)));
                    return;
                }

                this_->set_data(hpx::util::unused);
                cud_debug.debug(debug::str<>("set data callback"), "event",
                    debug::ptr(this_));
            }

            // the runtime pointer is needed by the callback based future
            hpx::runtime* rt_;
            // the cuda event is needed by the event based future
            cudaEvent_t event_;
        };

        // -----------------------------------------------------------------
        // intrusive pointer for future_data
        using future_data_ptr =
            memory::intrusive_ptr<future_data<hpx::util::internal_allocator<>>>;

        // -----------------------------------------------------------------
        // we track requests and future data in two vectors even though
        // we have the request stored in the future data already
        // the reason for this is because we can use MPI_Testany
        // with a vector of requests to save overheads compared
        // to testing one by one every item using a list
        HPX_EXPORT std::vector<future_data_ptr>& get_active_futures();

        // -----------------------------------------------------------------
        // define a lockfree queue type to place requests in prior to handling
        // this is done only to avoid taking a lock every time a request is
        // returned from MPI. Instead the requests are placed into a queue
        // and the polling code pops them prior to calling Testany
        using queue_type = concurrency::ConcurrentQueue<future_data_ptr>;
        queue_type& get_event_queue();

        // -----------------------------------------------------------------
        // used internally to add an cuda_event_type to the lockfree queue
        // that will be used by the polling routines to check when requests
        // have completed
        HPX_EXPORT void add_to_event_queue(future_data_ptr data);

        // -----------------------------------------------------------------
        // used internally to add a request to the main polling vector/list
        // that is passed to MPI_Testany
        HPX_EXPORT void add_to_polling_list(future_data_ptr data);

        // -------------------------------------------------------------
        // main API call to get a future from a stream using allocator
        template <typename Allocator>
        hpx::future<void> get_future_with_callback(
            Allocator const& a, cudaStream_t stream)
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

            traits::construct(alloc, p.get(), init_no_addref{}, alloc, stream,
                callback_mode{});

            return hpx::traits::future_access<future<void>>::create(
                p.release(), false);
        }

        // -------------------------------------------------------------
        // main API call to get a future from a stream using allocator
        template <typename Allocator>
        hpx::future<void> get_future_with_event(
            Allocator const& a, cudaStream_t stream)
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
                alloc, p.get(), init_no_addref{}, alloc, stream, event_mode{});

            // queue the future state internally for processing
            detail::add_to_event_queue(p.get());

            return hpx::traits::future_access<future<void>>::create(
                p.release(), false);
        }

        // -------------------------------------------------------------
        // non allocator version of : get future with a callback set
        HPX_EXPORT hpx::future<void> get_future_with_callback(cudaStream_t);

        // -------------------------------------------------------------
        // non allocator version of : get future with an event set
        HPX_EXPORT hpx::future<void> get_future_with_event(cudaStream_t);

        // -------------------------------------------------------------
        void register_polling(hpx::threads::thread_pool_base& pool);
        void unregister_polling(hpx::threads::thread_pool_base& pool);

        // -------------------------------------------------------------
    }    // namespace detail

    // -----------------------------------------------------------------
    // This RAII helper class enables polling for a scoped block
    struct HPX_NODISCARD enable_user_polling
    {
        enable_user_polling(std::string const& pool_name = "")
          : pool_name_(pool_name)
        {
            // install polling loop on requested thread pool
            if (pool_name_.empty())
            {
                detail::register_polling(hpx::resource::get_thread_pool(0));
            }
            else
            {
                detail::register_polling(
                    hpx::resource::get_thread_pool(pool_name_));
            }
        }

        ~enable_user_polling()
        {
            if (pool_name_.empty())
            {
                detail::unregister_polling(hpx::resource::get_thread_pool(0));
            }
            else
            {
                detail::unregister_polling(
                    hpx::resource::get_thread_pool(pool_name_));
            }
        }

    private:
        std::string pool_name_;
    };

}}}    // namespace hpx::cuda::experimental
