//  Copyright (c) 2020 John Biddiscombe
//  Copyright (c) 2016 Thomas Heller
//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/basic_execution.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/memory.hpp>
#include <hpx/threading_base.hpp>

#include <cuda_runtime.h>

#include <cstddef>
#include <iosfwd>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace hpx { namespace cuda {

    using print_on = debug::enable_print<false>;
    static constexpr print_on cuda_debug("CUDAFUT");

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
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
        // main API call to get a future from a stream using allocator
        template <typename Allocator>
        hpx::future<void> get_future(Allocator const& a, cudaStream_t stream)
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

            traits::construct(alloc, p.get(), init_no_addref{}, alloc, stream);

            return hpx::traits::future_access<future<void>>::create(
                p.release(), false);
        }

        // -------------------------------------------------------------
        // main API call to get a future from a stream
        HPX_EXPORT hpx::future<void> get_future(cudaStream_t);

        // -------------------------------------------------------------
        // cuda future data implementation
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
            }

            future_data()
              : rt_(hpx::get_runtime_ptr())
            {
            }

            future_data(init_no_addref no_addref, other_allocator const& alloc,
                cudaStream_t stream)
              : lcos::detail::future_data_allocator<void, Allocator>(
                    no_addref, alloc)
              , rt_(hpx::get_runtime_ptr())
            {
                init(stream);
            }

            void init(cudaStream_t stream)
            {
                // Hold on to the shared state on behalf of the cuda runtime
                // right away as the callback could be called immediately.
                lcos::detail::intrusive_ptr_add_ref(this);

                cudaError_t error =
                    cudaStreamAddCallback(stream, stream_callback, this, 0);
                if (error != cudaSuccess)
                {
                    // callback was not called, release object
                    lcos::detail::intrusive_ptr_release(this);

                    // report error
                    HPX_THROW_EXCEPTION(kernel_error,
                        "cuda::detail::future_data::future_data()",
                        std::string("cudaStreamAddCallback failed: ") +
                            cudaGetErrorString(error));
                }
            }

        private:
            hpx::runtime* rt_;
        };
    }    // namespace detail
}}       // namespace hpx::cuda
