///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#ifndef HPX_COMPUTE_CUDA_TARGET_HPP
#define HPX_COMPUTE_CUDA_TARGET_HPP

#include <hpx/config.hpp>

#if defined(HPX_HAVE_CUDA)
#include <hpx/exception.hpp>

#include <cuda_runtime.h>

#include <string>
#include <utility>

namespace hpx { namespace compute { namespace cuda
{
    struct target
    {
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
            error = cudaStreamCreate(&handle_.stream_);
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
            error = cudaStreamCreate(&handle_.stream_);
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

    private:
        native_handle_type handle_;
    };
}}}

#endif
#endif
