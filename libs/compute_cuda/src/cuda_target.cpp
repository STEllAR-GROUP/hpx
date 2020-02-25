//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if defined(HPX_HAVE_CUDA)

#include <hpx/allocator_support/internal_allocator.hpp>
#include <hpx/assertion.hpp>
#include <hpx/compute/cuda/target.hpp>
#include <hpx/errors.hpp>
#include <hpx/memory/intrusive_ptr.hpp>
#include <hpx/runtime/find_here.hpp>
#include <hpx/runtime/naming/id_type_impl.hpp>
#include <hpx/runtime_fwd.hpp>
#include <hpx/threading_base/thread_helpers.hpp>
#include <hpx/traits/future_access.hpp>

#if !defined(HPX_COMPUTE_DEVICE_CODE)
#if defined(HPX_HAVE_MORE_THAN_64_THREADS)
#if defined(HPX_HAVE_MAX_CPU_COUNT)
#include <hpx/serialization/bitset.hpp>
#else
#include <hpx/serialization/dynamic_bitset.hpp>
#endif
#endif
#include <hpx/serialization/serialize.hpp>
#endif

#include <cstddef>
#include <memory>
#include <string>
#include <utility>

#include <cuda_runtime.h>

namespace hpx { namespace compute { namespace cuda {
    namespace detail {
        runtime_registration_wrapper::runtime_registration_wrapper(
            hpx::runtime* rt)
          : rt_(rt)
        {
            HPX_ASSERT(rt);

            // Register this thread with HPX, this should be done once for
            // each external OS-thread intended to invoke HPX functionality.
            // Calling this function more than once on the same thread will
            // report an error.
            hpx::error_code ec(hpx::lightweight);    // ignore errors
            hpx::register_thread(rt_, "cuda", ec);
        }

        runtime_registration_wrapper::~runtime_registration_wrapper()
        {
            // Unregister the thread from HPX, this should be done once in
            // the end before the external thread exists.
            hpx::unregister_thread(rt_);
        }

        hpx::future<void> get_future(cudaStream_t stream)
        {
            return get_future(hpx::util::internal_allocator<>{}, stream);
        }
    }    // namespace detail

    void target::native_handle_type::init_processing_units()
    {
        cudaDeviceProp props;
        cudaError_t error = cudaGetDeviceProperties(&props, device_);
        if (error != cudaSuccess)
        {
            // report error
            HPX_THROW_EXCEPTION(kernel_error,
                "cuda::default_executor::processing_units_count()",
                std::string("cudaGetDeviceProperties failed: ") +
                    cudaGetErrorString(error));
        }

        std::size_t mp = props.multiProcessorCount;
        switch (props.major)
        {
        case 2:    // Fermi
            if (props.minor == 1)
            {
                mp = mp * 48;
            }
            else
            {
                mp = mp * 32;
            }
            break;
        case 3:    // Kepler
            mp = mp * 192;
            break;
        case 5:    // Maxwell
            mp = mp * 128;
            break;
        case 6:    // Pascal
            mp = mp * 64;
            break;
        default:
            break;
        }
        processing_units_ = mp;
        processor_family_ = props.major;
        processor_name_ = props.name;
    }

    target::native_handle_type::native_handle_type(int device)
      : device_(device)
      , stream_(0)
    {
        init_processing_units();
    }

    target::native_handle_type::~native_handle_type()
    {
        reset();
    }

    void target::native_handle_type::reset() noexcept
    {
        if (stream_)
            cudaStreamDestroy(stream_);    // ignore error
    }

    target::native_handle_type::native_handle_type(
        target::native_handle_type const& rhs) noexcept
      : device_(rhs.device_)
      , processing_units_(rhs.processing_units_)
      , processor_family_(rhs.processor_family_)
      , processor_name_(rhs.processor_name_)
      , stream_(0)
    {
    }

    target::native_handle_type::native_handle_type(
        target::native_handle_type&& rhs) noexcept
      : device_(rhs.device_)
      , processing_units_(rhs.processing_units_)
      , processor_family_(rhs.processor_family_)
      , processor_name_(rhs.processor_name_)
      , stream_(rhs.stream_)
    {
        rhs.stream_ = 0;
    }

    target::native_handle_type& target::native_handle_type::operator=(
        target::native_handle_type const& rhs) noexcept
    {
        if (this == &rhs)
            return *this;

        device_ = rhs.device_;
        processing_units_ = rhs.processing_units_;
        processor_family_ = rhs.processor_family_;
        processor_name_ = rhs.processor_name_;
        reset();

        return *this;
    }

    target::native_handle_type& target::native_handle_type::operator=(
        target::native_handle_type&& rhs) noexcept
    {
        if (this == &rhs)
            return *this;

        device_ = rhs.device_;
        processing_units_ = rhs.processing_units_;
        processor_family_ = rhs.processor_family_;
        processor_name_ = rhs.processor_name_;
        stream_ = rhs.stream_;
        rhs.stream_ = 0;

        return *this;
    }

    cudaStream_t target::native_handle_type::get_stream() const
    {
        std::lock_guard<mutex_type> l(mtx_);

        if (stream_ == 0)
        {
            cudaError_t error = cudaSetDevice(device_);
            if (error != cudaSuccess)
            {
                HPX_THROW_EXCEPTION(kernel_error,
                    "cuda::target::native_handle::get_stream()",
                    std::string("cudaSetDevice failed: ") +
                        cudaGetErrorString(error));
            }
            error = cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
            if (error != cudaSuccess)
            {
                HPX_THROW_EXCEPTION(kernel_error,
                    "cuda::target::native_handle::get_stream()",
                    std::string("cudaStreamCreate failed: ") +
                        cudaGetErrorString(error));
            }
        }
        return stream_;
    }

    ///////////////////////////////////////////////////////////////////////////
    void target::synchronize() const
    {
        // FIXME: implement remote targets
        HPX_ASSERT(hpx::find_here() == locality_);

        cudaStream_t stream = handle_.get_stream();

        if (stream == 0)
        {
            HPX_THROW_EXCEPTION(invalid_status, "cuda::target::synchronize",
                "no stream available");
        }

        cudaError_t error = cudaStreamSynchronize(stream);
        if (error != cudaSuccess)
        {
            HPX_THROW_EXCEPTION(kernel_error, "cuda::target::synchronize",
                std::string("cudaStreamSynchronize failed: ") +
                    cudaGetErrorString(error));
        }
    }

    hpx::future<void> target::get_future() const
    {
        return detail::get_future(handle_.get_stream());
    }

    target& get_default_target()
    {
        static target target_;
        return target_;
    }

#if !defined(HPX_COMPUTE_DEVICE_CODE)
    ///////////////////////////////////////////////////////////////////////////
    void target::serialize(
        serialization::input_archive& ar, const unsigned int version)
    {
        ar >> handle_.device_ >> locality_;
    }

    void target::serialize(
        serialization::output_archive& ar, const unsigned int version)
    {
        ar << handle_.device_ << locality_;
    }
#endif
}}}    // namespace hpx::compute::cuda

#endif
