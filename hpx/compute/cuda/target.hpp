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
#include <hpx/compute/cuda/get_targets.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/runtime/find_here.hpp>
#include <hpx/runtime/runtime_fwd.hpp>

#if !defined(__CUDA_ARCH__)
#include <hpx/runtime/serialization/serialization_fwd.hpp>
#endif
#include <cuda_runtime.h>

#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace compute { namespace cuda
{
    ///////////////////////////////////////////////////////////////////////////
    struct target
    {
    public:
        struct HPX_EXPORT native_handle_type
        {
            typedef hpx::lcos::local::spinlock mutex_type;

            native_handle_type(int device = 0);

            ~native_handle_type();

            native_handle_type(native_handle_type const& rhs) HPX_NOEXCEPT;
            native_handle_type(native_handle_type && rhs) HPX_NOEXCEPT;

            native_handle_type& operator=(native_handle_type const& rhs) HPX_NOEXCEPT;
            native_handle_type& operator=(native_handle_type && rhs) HPX_NOEXCEPT;

            cudaStream_t get_stream() const;

            int get_device() const HPX_NOEXCEPT
            {
                return device_;
            }

            void reset() HPX_NOEXCEPT;

        private:
            friend struct target;

            mutable mutex_type mtx_;
            int device_;
            mutable cudaStream_t stream_;
        };

        // Constructs default target
        target()
          : handle_(), locality_(hpx::find_here())
        {}

        // Constructs target from a given device ID
        explicit target(int device)
          : handle_(device), locality_(hpx::find_here())
        {}

        explicit target(hpx::id_type const& locality, int device)
          : handle_(device), locality_(locality)
        {}

        target(target const& rhs) HPX_NOEXCEPT
          : handle_(rhs.handle_),
            locality_(rhs.locality_)
        {}

        target(target && rhs) HPX_NOEXCEPT
          : handle_(std::move(rhs.handle_)),
            locality_(std::move(rhs.locality_))
        {}

        target& operator=(target const& rhs) HPX_NOEXCEPT
        {
            if (&rhs != this)
            {
                handle_ = rhs.handle_;
                locality_ = rhs.locality_;
            }
            return *this;
        }

        target& operator=(target && rhs) HPX_NOEXCEPT
        {
            if (&rhs != this)
            {
                handle_ = std::move(rhs.handle_);
                locality_ = std::move(rhs.locality_);
            }
            return *this;
        }

        native_handle_type& native_handle() HPX_NOEXCEPT
        {
            return handle_;
        }
        native_handle_type const& native_handle() const HPX_NOEXCEPT
        {
            return handle_;
        }

        hpx::id_type const& get_locality() const HPX_NOEXCEPT
        {
            return locality_;
        }

        void synchronize() const;

        hpx::future<void> get_future() const;

        static std::vector<target> get_local_targets()
        {
            return cuda::get_local_targets();
        }
        static hpx::future<std::vector<target> >
            get_targets(hpx::id_type const& locality)
        {
            return cuda::get_targets(locality);
        }

        friend bool operator==(target const& lhs, target const& rhs)
        {
            return lhs.handle_.get_device() == rhs.handle_.get_device() &&
                lhs.locality_ == rhs.locality_;
        }

    private:
#if !defined(__CUDA_ARCH__)
        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive& ar, const unsigned int version)
        {
            ar & handle_.device_ & locality_;
        }
#endif
        native_handle_type handle_;
        hpx::id_type locality_;
    };

    HPX_API_EXPORT target& get_default_target();
}}}

#include <hpx/config/warnings_suffix.hpp>

#endif
#endif
