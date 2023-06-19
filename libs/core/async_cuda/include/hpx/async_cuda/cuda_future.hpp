//  Copyright (c) 2020 John Biddiscombe
//  Copyright (c) 2016 Thomas Heller
//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/async_cuda/cuda_event.hpp>
#include <hpx/async_cuda/cuda_exception.hpp>
#include <hpx/async_cuda/custom_gpu_api.hpp>
#include <hpx/async_cuda/detail/cuda_debug.hpp>
#include <hpx/async_cuda/detail/cuda_event_callback.hpp>
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

namespace hpx { namespace cuda { namespace experimental {
    using event_mode = std::true_type;
    using callback_mode = std::false_type;

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        // -------------------------------------------------------------
        // cuda future data implementation
        // This version supports 2 modes of operation
        // 1) a callback based future that is made ready
        // by a cuda callback when the stream event occurs
        // 2) an event based callback that must be polled/queried by
        // the runtime to set the future ready state
        template <typename Allocator, typename Mode>
        struct future_data;

        // -------------------------------------------------------------
        // helper struct to delete future data in destructor
        template <typename Allocator>
        struct release_on_exit
        {
            explicit release_on_exit(
                future_data<Allocator, callback_mode>* data)
              : data_(data)
            {
            }

            ~release_on_exit()
            {
                // release the shared state
                lcos::detail::intrusive_ptr_release(data_);
            }

            future_data<Allocator, callback_mode>* data_;
        };

        template <typename Allocator>
        struct future_data<Allocator, event_mode>
          : lcos::detail::future_data_allocator<void, Allocator,
                future_data<Allocator, event_mode>>
        {
            HPX_NON_COPYABLE(future_data);

            using init_no_addref =
                typename lcos::detail::future_data_allocator<void, Allocator,
                    future_data>::init_no_addref;

            using other_allocator = typename std::allocator_traits<
                Allocator>::template rebind_alloc<future_data>;

            future_data() {}

            future_data(init_no_addref no_addref, other_allocator const& alloc,
                cudaStream_t stream, int device)
              : lcos::detail::future_data_allocator<void, Allocator,
                    future_data>(no_addref, alloc)
            {
                add_event_callback(
                    [fdp = hpx::intrusive_ptr<future_data>(this)](
                        cudaError_t status) {
                        HPX_ASSERT(status != cudaErrorNotReady);

                        if (status == cudaSuccess)
                        {
                            fdp->set_data(hpx::util::unused);
                        }
                        else
                        {
                            fdp->set_exception(
                                std::make_exception_ptr(cuda_exception(
                                    std::string(
                                        "cuda function returned error code :") +
                                        cudaGetErrorString(status),
                                    status)));
                        }
                    },
                    stream, device);
            }
        };

        template <typename Allocator>
        struct future_data<Allocator, callback_mode>
          : lcos::detail::future_data_allocator<void, Allocator,
                future_data<Allocator, callback_mode>>
        {
            HPX_NON_COPYABLE(future_data);

            using init_no_addref =
                typename lcos::detail::future_data_allocator<void, Allocator,
                    future_data>::init_no_addref;

            using other_allocator = typename std::allocator_traits<
                Allocator>::template rebind_alloc<future_data>;

            future_data()
              : rt_(hpx::get_runtime_ptr())
            {
            }

            future_data(init_no_addref no_addref, other_allocator const& alloc,
                cudaStream_t stream, int device)
              : lcos::detail::future_data_allocator<void, Allocator,
                    future_data>(no_addref, alloc)
              , rt_(hpx::get_runtime_ptr())
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
                cudaStream_t, cudaError_t error, void* user_data)
            {
                future_data* this_ = static_cast<future_data*>(user_data);

                release_on_exit<Allocator> on_exit(this_);

                if (error != cudaSuccess)
                {
                    this_->set_exception(
                        HPX_GET_EXCEPTION(hpx::error::kernel_error,
                            "cuda::detail::future_data::stream_callback()",
                            std::string("cudaStreamAddCallback failed: ") +
                                cudaGetErrorString(error)));
                    return;
                }

                this_->set_data(hpx::util::unused);
                cud_debug.debug(debug::str<>("set data callback"), "event",
                    debug::ptr(this_));
            }

            hpx::runtime* rt_;
        };

        // -------------------------------------------------------------
        // main API call to get a future from a stream using allocator, and the
        // specified mode
        template <typename Allocator, typename Mode>
        hpx::future<void> get_future(Allocator const& a, cudaStream_t stream, int device = 0)
        {
            using shared_state = future_data<Allocator, Mode>;

            using other_allocator = typename std::allocator_traits<
                Allocator>::template rebind_alloc<shared_state>;
            using traits = std::allocator_traits<other_allocator>;

            using init_no_addref = typename shared_state::init_no_addref;

            using unique_ptr = std::unique_ptr<shared_state,
                util::allocator_deleter<other_allocator>>;

            other_allocator alloc(a);
            unique_ptr p(traits::allocate(alloc, 1),
                hpx::util::allocator_deleter<other_allocator>{alloc});

            traits::construct(alloc, p.get(), init_no_addref{}, alloc, stream, device);

            return hpx::traits::future_access<future<void>>::create(
                p.release(), false);
        }

        // -------------------------------------------------------------
        // main API call to get a future from a stream using allocator
        template <typename Allocator>
        hpx::future<void> get_future_with_callback(
            Allocator const& a, cudaStream_t stream)
        {
            return get_future<Allocator, callback_mode>(a, stream, 0);
        }

        // -------------------------------------------------------------
        // main API call to get a future from a stream using allocator
        template <typename Allocator>
        hpx::future<void> get_future_with_event(
            Allocator const& a, cudaStream_t stream, int device)
        {
            return get_future<Allocator, event_mode>(a, stream, device);
        }

        // -------------------------------------------------------------
        // non allocator version of : get future with a callback set
        HPX_CORE_EXPORT hpx::future<void> get_future_with_callback(
            cudaStream_t);

        // -------------------------------------------------------------
        // non allocator version of : get future with an event set
        HPX_CORE_EXPORT hpx::future<void> get_future_with_event(cudaStream_t, int);
    }    // namespace detail
}}}      // namespace hpx::cuda::experimental

namespace hpx { namespace traits { namespace detail {

    template <typename Allocator, typename Mode, typename NewAllocator>
    struct shared_state_allocator<
        hpx::cuda::experimental::detail::future_data<Allocator, Mode>,
        NewAllocator>
    {
        using type =
            hpx::cuda::experimental::detail::future_data<NewAllocator, Mode>;
    };
}}}    // namespace hpx::traits::detail
