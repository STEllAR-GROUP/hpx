///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Thomas Heller
//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#ifndef HPX_COMPUTE_CUDA_TARGET_HPP
#define HPX_COMPUTE_CUDA_TARGET_HPP

#include <hpx/config.hpp>

#if defined(HPX_HAVE_CUDA)
#include <hpx/allocator_support/allocator_deleter.hpp>
#include <hpx/compute/cuda/get_targets.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/runtime/find_here.hpp>
#include <hpx/runtime/runtime_fwd.hpp>
#include <hpx/synchronization/spinlock.hpp>
#include <hpx/traits/future_access.hpp>

#include <hpx/serialization/serialization_fwd.hpp>

#include <cuda_runtime.h>

#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace compute { namespace cuda {

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {

        HPX_API_EXPORT hpx::future<void> get_future(cudaStream_t);

        template <typename Allocator>
        hpx::future<void> get_future(Allocator const& a, cudaStream_t stream)
        {
            using shared_state =
                typename traits::detail::shared_state_allocator<
                    lcos::detail::future_data, Allocator>::type;

            using other_allocator =
                typename std::allocator_traits<Allocator>::
                    template rebind_alloc<shared_state>;
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
    }

    ///////////////////////////////////////////////////////////////////////////
    struct target
    {
    public:
        struct HPX_EXPORT native_handle_type
        {
            typedef hpx::lcos::local::spinlock mutex_type;

            HPX_HOST_DEVICE native_handle_type(int device = 0);

            HPX_HOST_DEVICE ~native_handle_type();

            HPX_HOST_DEVICE native_handle_type(
                native_handle_type const& rhs) noexcept;
            HPX_HOST_DEVICE native_handle_type(
                native_handle_type&& rhs) noexcept;

            HPX_HOST_DEVICE native_handle_type& operator=(
                native_handle_type const& rhs) noexcept;
            HPX_HOST_DEVICE native_handle_type& operator=(
                native_handle_type&& rhs) noexcept;

            HPX_HOST_DEVICE cudaStream_t get_stream() const;

            HPX_HOST_DEVICE int get_device() const noexcept
            {
                return device_;
            }

            HPX_HOST_DEVICE std::size_t processing_units() const
            {
                return processing_units_;
            }

            HPX_HOST_DEVICE std::size_t processor_family() const
            {
                return processor_family_;
            }

            HPX_HOST_DEVICE std::string processor_name() const
            {
                return processor_name_;
            }

            void reset() noexcept;

        private:
            void init_processing_units();
            friend struct target;

            mutable mutex_type mtx_;
            int device_;
            std::size_t processing_units_;
            std::size_t processor_family_;
            std::string processor_name_;
            mutable cudaStream_t stream_;
        };

        // Constructs default target
        HPX_HOST_DEVICE target()
          : handle_()
#if !defined(HPX_COMPUTE_DEVICE_CODE)
          , locality_(hpx::find_here())
#endif
        {
        }

        // Constructs target from a given device ID
        explicit HPX_HOST_DEVICE target(int device)
          : handle_(device)
#if !defined(HPX_COMPUTE_DEVICE_CODE)
          , locality_(hpx::find_here())
#endif
        {
        }

        HPX_HOST_DEVICE target(hpx::id_type const& locality, int device)
          : handle_(device)
#if !defined(HPX_COMPUTE_DEVICE_CODE)
          , locality_(locality)
#endif
        {
        }

        HPX_HOST_DEVICE target(target const& rhs) noexcept
          : handle_(rhs.handle_)
#if !defined(HPX_COMPUTE_DEVICE_CODE)
          , locality_(rhs.locality_)
#endif
        {
        }

        HPX_HOST_DEVICE target(target&& rhs) noexcept
          : handle_(std::move(rhs.handle_))
#if !defined(HPX_COMPUTE_DEVICE_CODE)
          , locality_(std::move(rhs.locality_))
#endif
        {
        }

        HPX_HOST_DEVICE target& operator=(target const& rhs) noexcept
        {
            if (&rhs != this)
            {
                handle_ = rhs.handle_;
#if !defined(HPX_COMPUTE_DEVICE_CODE)
                locality_ = rhs.locality_;
#endif
            }
            return *this;
        }

        HPX_HOST_DEVICE target& operator=(target&& rhs) noexcept
        {
            if (&rhs != this)
            {
                handle_ = std::move(rhs.handle_);
#if !defined(HPX_COMPUTE_DEVICE_CODE)
                locality_ = std::move(rhs.locality_);
#endif
            }
            return *this;
        }

        HPX_HOST_DEVICE
        native_handle_type& native_handle() noexcept
        {
            return handle_;
        }
        HPX_HOST_DEVICE
        native_handle_type const& native_handle() const noexcept
        {
            return handle_;
        }

        HPX_HOST_DEVICE void synchronize() const;

        HPX_HOST_DEVICE hpx::id_type const& get_locality() const noexcept
        {
            return locality_;
        }

        hpx::future<void> get_future() const;

        template <typename Allocator>
        hpx::future<void> get_future(Allocator const& alloc) const
        {
            return detail::get_future(alloc, handle_.get_stream());
        }

        static std::vector<target> get_local_targets()
        {
            return cuda::get_local_targets();
        }
        static hpx::future<std::vector<target>> get_targets(
            hpx::id_type const& locality)
        {
            return cuda::get_targets(locality);
        }

        friend bool operator==(target const& lhs, target const& rhs)
        {
            return lhs.handle_.get_device() == rhs.handle_.get_device() &&
                lhs.locality_ == rhs.locality_;
        }

    private:
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        friend class hpx::serialization::access;

        void serialize(serialization::input_archive& ar, const unsigned int);
        void serialize(serialization::output_archive& ar, const unsigned int);
#endif

        native_handle_type handle_;
        hpx::id_type locality_;
    };

    using detail::get_future;
    HPX_API_EXPORT target& get_default_target();
}}}    // namespace hpx::compute::cuda

#include <hpx/config/warnings_suffix.hpp>

#endif
#endif
