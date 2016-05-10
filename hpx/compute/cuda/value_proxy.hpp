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
            traits::access_target<target> access_target;

    public:
        value_proxy(T *p, target & tgt)
          : p_(p)
          , tgt_(tgt)
        {}

        value_proxy& operator=(T const& t)
        {
            access_target::write(tgt_, p_, &t);
            return *this;
        }

        operator T() const
        {
            return access_target::read(tgt_, p_);
        }

        T* device_ptr()
        {
            return p_;
        }

    private:
        T* p_;
        target& tgt_;
    };

    template <typename T>
    class value_proxy<T const>
    {
        typedef
            traits::access_target<target> access_target;

    public:
        value_proxy(T *p, target & tgt)
          : p_(p)
          , tgt_(tgt)
        {}

        operator T() const
        {
            return access_target::read(tgt_, p_);
        }

        T* device_ptr()
        {
            return p_;
        }

    private:
        T* p_;
        target& tgt_;
    };

}}}

#endif
