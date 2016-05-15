///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#ifndef HPX_COMPUTE_CUDA_VALUE_PROXY_HPP
#define HPX_COMPUTE_CUDA_VALUE_PROXY_HPP

#include <hpx/compute/cuda/target.hpp>
#include <hpx/compute/cuda/traits/access_target.hpp>

namespace hpx { namespace compute { namespace cuda
{
    template <typename T>
    class value_proxy
    {
        typedef
            traits::access_target<cuda::target> access_target;

    public:
        value_proxy(T *p, cuda::target & tgt) HPX_NOEXCEPT
          : p_(p)
          , target_(tgt)
        {}

        value_proxy& operator=(T const& t)
        {
            access_target::write(target_, p_, &t);
            return *this;
        }

        operator T() const
        {
            return access_target::read(target_, p_);
        }

        T* device_ptr() const HPX_NOEXCEPT
        {
            return p_;
        }

        cuda::target& target() const HPX_NOEXCEPT
        {
            return target_;
        }

    private:
        T* p_;
        cuda::target& target_;
    };

    template <typename T>
    class value_proxy<T const>
    {
        typedef
            traits::access_target<cuda::target> access_target;

    public:
        value_proxy(T *p, cuda::target & tgt) HPX_NOEXCEPT
          : p_(p)
          , target_(tgt)
        {}

        value_proxy(value_proxy<T> const& other)
          : p_(other.device_ptr())
          , target_(other.target())
        {}

        operator T() const
        {
            return access_target::read(target_, p_);
        }

        T* device_ptr() const HPX_NOEXCEPT
        {
            return p_;
        }

        cuda::target& target() const HPX_NOEXCEPT
        {
            return target_;
        }

    private:
        T* p_;
        cuda::target& target_;
    };
}}}

#endif
