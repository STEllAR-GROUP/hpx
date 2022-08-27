///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Thomas Heller
//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <hpx/config.hpp>

#include <hpx/allocator_support/allocator_deleter.hpp>
#include <hpx/assert.hpp>
#include <hpx/async_cuda/cuda_future.hpp>
#include <hpx/async_cuda/get_targets.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/futures/traits/future_access.hpp>
#include <hpx/synchronization/spinlock.hpp>
#include <hpx/type_support/unused.hpp>

#include <hpx/async_cuda/custom_gpu_api.hpp>

#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace cuda { namespace experimental {

    ///////////////////////////////////////////////////////////////////////////
    struct target
    {
    public:
        struct HPX_CORE_EXPORT native_handle_type
        {
            typedef hpx::lcos::local::spinlock mutex_type;

            native_handle_type(int device = 0);

            ~native_handle_type();

            native_handle_type(native_handle_type const& rhs) noexcept;
            native_handle_type(native_handle_type&& rhs) noexcept;

            native_handle_type& operator=(
                native_handle_type const& rhs) noexcept;
            native_handle_type& operator=(native_handle_type&& rhs) noexcept;

            cudaStream_t get_stream() const;

            int get_device() const noexcept
            {
                return device_;
            }

            std::size_t processing_units() const
            {
                return processing_units_;
            }

            std::size_t processor_family() const
            {
                return processor_family_;
            }

            std::string processor_name() const
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
        {
        }

        // Constructs target from a given device ID
        explicit HPX_HOST_DEVICE target(int device)
          : handle_(device)
        {
        }

        HPX_HOST_DEVICE target(target const& rhs) noexcept
          : handle_(rhs.handle_)
        {
        }

        HPX_HOST_DEVICE target(target&& rhs) noexcept
          : handle_(HPX_MOVE(rhs.handle_))
        {
        }

        HPX_HOST_DEVICE target& operator=(target const& rhs) noexcept
        {
            if (&rhs != this)
            {
                handle_ = rhs.handle_;
            }
            return *this;
        }

        HPX_HOST_DEVICE target& operator=(target&& rhs) noexcept
        {
            if (&rhs != this)
            {
                handle_ = HPX_MOVE(rhs.handle_);
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

        void synchronize() const;

        hpx::future<void> get_future_with_event() const;
        hpx::future<void> get_future_with_callback() const;

        template <typename Allocator>
        hpx::future<void> get_future_with_event(Allocator const& alloc) const
        {
            return detail::get_future_with_event(alloc, handle_.get_stream());
        }

        template <typename Allocator>
        hpx::future<void> get_future_with_callback(Allocator const& alloc) const
        {
            return detail::get_future_with_callback(
                alloc, handle_.get_stream());
        }

        static std::vector<target> get_local_targets()
        {
            return cuda::experimental::get_local_targets();
        }

        friend bool operator==(target const& lhs, target const& rhs)
        {
            return lhs.handle_.get_device() == rhs.handle_.get_device();
        }

    private:
        native_handle_type handle_;
    };

    using detail::get_future_with_callback;
    HPX_CORE_EXPORT target& get_default_target();
}}}    // namespace hpx::cuda::experimental

#include <hpx/config/warnings_suffix.hpp>
