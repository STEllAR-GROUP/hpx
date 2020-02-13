///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_CUDA)

#include <hpx/algorithms/traits/is_value_proxy.hpp>
#include <hpx/compute/cuda/traits/access_target.hpp>
#include <hpx/cuda_support/target.hpp>

#include <type_traits>

namespace hpx { namespace compute { namespace cuda {
    template <typename T>
    class value_proxy
    {
        typedef traits::access_target<hpx::cuda::target> access_target;

    public:
        value_proxy(T* p, hpx::cuda::target& tgt) noexcept
          : p_(p)
          , target_(&tgt)
        {
        }

        value_proxy(value_proxy const& other)
          : p_(other.p_)
          , target_(other.target_)
        {
        }

        value_proxy& operator=(T const& t)
        {
            access_target::write(*target_, p_, &t);
            return *this;
        }

        value_proxy& operator=(value_proxy const& other)
        {
            p_ = other.p_;
            target_ = other.target_;
            return *this;
        }

        // Note: The difference of signature allows to define proper
        //       implicit casts for device code and host code

        HPX_DEVICE operator T&()
        {
            return *p_;
        }

        HPX_HOST operator T() const
        {
            return access_target::read(*target_, p_);
        }

        T* operator&() const
        {
            return p_;
        }

        T* device_ptr() const noexcept
        {
            return p_;
        }

        hpx::cuda::target& target() const noexcept
        {
            return *target_;
        }

    private:
        T* p_;
        hpx::cuda::target* target_;
    };

    template <typename T>
    class value_proxy<T const>
    {
        typedef traits::access_target<hpx::cuda::target> access_target;

    public:
        typedef T const proxy_type;

        value_proxy(T* p, hpx::cuda::target& tgt) noexcept
          : p_(p)
          , target_(tgt)
        {
        }

        value_proxy(value_proxy<T> const& other)
          : p_(other.device_ptr())
          , target_(other.target())
        {
        }

        HPX_HOST_DEVICE operator T() const
        {
            return access_target::read(target_, p_);
        }

        T* device_ptr() const noexcept
        {
            return p_;
        }

        hpx::cuda::target& target() const noexcept
        {
            return target_;
        }

    private:
        T* p_;
        hpx::cuda::target& target_;
    };
}}}    // namespace hpx::compute::cuda

namespace hpx { namespace traits {
    template <typename T>
    struct is_value_proxy<hpx::compute::cuda::value_proxy<T>> : std::true_type
    {
    };
}}    // namespace hpx::traits

#endif
