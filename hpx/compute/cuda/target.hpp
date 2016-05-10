///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Thomas Heller
//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#ifndef HPX_COMPUTE_CUDA_TARGET_HPP
#define HPX_COMPUTE_CUDA_TARGET_HPP

#include <hpx/config.hpp>

#if defined(HPX_HAVE_CUDA)
#include <hpx/exception.hpp>
#include <hpx/runtime_fwd.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/util/assert.hpp>

#include <cuda_runtime.h>

#include <string>
#include <utility>

namespace hpx { namespace compute { namespace cuda
{
    namespace detail
    {
        struct runtime_registration_wrapper
        {
            runtime_registration_wrapper(hpx::runtime* rt)
              : rt_(rt)
            {
                HPX_ASSERT(rt);

                // Register this thread with HPX, this should be done once for
                // each external OS-thread intended to invoke HPX functionality.
                // Calling this function more than once will silently fail
                // (will return false).
                hpx::register_thread(rt_, "cuda");
            }
            ~runtime_registration_wrapper()
            {
                // Unregister the thread from HPX, this should be done once in
                // the end before the external thread exists.
                hpx::unregister_thread(rt_);
            }

            hpx::runtime* rt_;
        };

        struct future_data;

        struct release_on_exit
        {
            release_on_exit(future_data* data)
              : data_(data)
            {}

            ~release_on_exit();

            future_data* data_;
        };

        struct future_data : lcos::detail::future_data<void>
        {
        private:
            static void CUDART_CB stream_callback(cudaStream_t stream,
                cudaError_t error, void* user_data)
            {
                future_data* this_ = static_cast<future_data*>(user_data);

                release_on_exit on_exit(this_);
                runtime_registration_wrapper wrap(this_->rt_);

                if (error != cudaSuccess)
                {
                    this_->set_exception(
                        HPX_GET_EXCEPTION(kernel_error,
                            "cuda::detail::future_data::stream_callback()",
                            std::string("cudaStreamAddCallback failed: ") +
                                cudaGetErrorString(error))
                    );
                    return;
                }

                this_->set_data(hpx::util::unused);
            }

        public:
            future_data(cudaStream_t stream)
              : rt_(hpx::get_runtime_ptr())
            {
                cudaError_t error = cudaStreamAddCallback(
                    stream, stream_callback, this, 0);
                if (error != cudaSuccess)
                {
                    HPX_THROW_EXCEPTION(kernel_error,
                        "cuda::detail::future_data::future_data()",
                        std::string("cudaStreamAddCallback failed: ") +
                            cudaGetErrorString(error));
                }

                // hold on to the shared state on behalf of the cuda runtime
                lcos::detail::intrusive_ptr_add_ref(this);
            }

        private:
            hpx::runtime* rt_;
        };

        ///////////////////////////////////////////////////////////////////////
        release_on_exit::~release_on_exit()
        {
            // release the shared state
            lcos::detail::intrusive_ptr_release(data_);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    struct target
    {
    private:
        HPX_MOVABLE_ONLY(target);

    public:
        struct native_handle_type
        {
            HPX_MOVABLE_ONLY(native_handle_type);

            native_handle_type(int device = 0)
              : device_(device), stream_(0)
            {}

            ~native_handle_type()
            {
                if (stream_)
                    cudaStreamDestroy(stream_);     // ignore error
            }

            native_handle_type(native_handle_type && rhs)
              : device_(rhs.device_), stream_(rhs.stream_)
            {
                rhs.stream_ = 0;
            }

            native_handle_type& operator=(native_handle_type && rhs)
            {
                device_ = rhs.device_;
                stream_ = rhs.stream_;
                rhs.stream_ = 0;
                return *this;
            }

            int device_;
            cudaStream_t stream_;
        };

        // Constructs default target
        target()
        {
            cudaError_t error = cudaSetDevice(handle_.device_);
            if (error != cudaSuccess)
            {
                HPX_THROW_EXCEPTION(kernel_error,
                    "cuda::target()",
                    std::string("cudaSetDevice failed: ") +
                        cudaGetErrorString(error));
            }
            error = cudaStreamCreateWithFlags(&handle_.stream_,
                cudaStreamNonBlocking);
            if (error != cudaSuccess)
            {
                HPX_THROW_EXCEPTION(kernel_error,
                    "cuda::target()",
                    std::string("cudaStreamCreate failed: ") +
                        cudaGetErrorString(error));
            }
        }

        // Constructs target from a given device ID
        explicit target(int device)
          : handle_(device)
        {
            cudaError_t error = cudaSetDevice(handle_.device_);
            if (error != cudaSuccess)
            {
                HPX_THROW_EXCEPTION(kernel_error,
                    "cuda::target()",
                    std::string("cudaSetDevice failed: ") +
                        cudaGetErrorString(error));
            }
            error = cudaStreamCreateWithFlags(&handle_.stream_,
                cudaStreamNonBlocking);
            if (error != cudaSuccess)
            {
                HPX_THROW_EXCEPTION(kernel_error,
                    "cuda::target()",
                    std::string("cudaStreamCreate failed: ") +
                        cudaGetErrorString(error));
            }
        }

        target(target && rhs)
          : handle_(std::move(rhs.handle_))
        {}

        target& operator=(target && rhs)
        {
            handle_ = std::move(rhs.handle_);
            return *this;
        }

        native_handle_type const& native_handle() const
        {
            return handle_;
        }

        void synchronize() const
        {
            cudaError_t error = cudaStreamSynchronize(handle_.stream_);
            if(error != cudaSuccess)
            {
                HPX_THROW_EXCEPTION(kernel_error,
                    "cuda::target::synchronize",
                    std::string("cudaStreamSynchronize failed: ") +
                        cudaGetErrorString(error));
            }
        }

        hpx::future<void> get_future() const
        {
            typedef detail::future_data shared_state_type;
            shared_state_type* p = new shared_state_type(handle_.stream_);
            return hpx::traits::future_access<hpx::future<void> >::create(p);
        }

    private:
        native_handle_type handle_;
    };
}}}

#endif
#endif
