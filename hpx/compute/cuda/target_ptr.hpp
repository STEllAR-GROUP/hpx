///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#ifndef HPX_COMPUTE_CUDA_TARGET_PTR_HPP
#define HPX_COMPUTE_CUDA_TARGET_PTR_HPP

#include <hpx/compute/cuda/target.hpp>
#include <hpx/compute/cuda/value_proxy.hpp>

#include <hpx/util/assert.hpp>

namespace hpx { namespace compute { namespace cuda
{
    template <typename T>
    class target_ptr
    {
    public:
        explicit target_ptr(std::nullptr_t)
          : p_(nullptr)
          , tgt_(nullptr)
        {}

        target_ptr(T *p, target & tgt)
          : p_(p)
          , tgt_(&tgt)
        {}

        target_ptr const& operator++()
        {
            HPX_ASSERT(p_);
            ++p_;
            return *this;
        }

        target_ptr const& operator--()
        {
            HPX_ASSERT(p_);
            --p_;
            return *this;
        }

        target_ptr operator++(int)
        {
            target_ptr tmp(*this);
            HPX_ASSERT(p_);
            ++p_;
            return tmp;
        }

        target_ptr operator--(int)
        {
            target_ptr tmp(*this);
            HPX_ASSERT(p_);
            --p_;
            return tmp;
        }

        explicit operator bool() const
        {
            return p_;
        }

        bool operator==(target_ptr const& other) const
        {
            return p_ == other.p_;
        }

        bool operator!=(target_ptr const& other) const
        {
            return p_ != other.p_;
        }

        bool operator<(target_ptr const& other) const
        {
            return p_ < other.p_;
        }

        bool operator>(target_ptr const& other) const
        {
            return p_ > other.p_;
        }

        bool operator<=(target_ptr const& other) const
        {
            return p_ <= other.p_;
        }

        bool operator>=(target_ptr const& other) const
        {
            return p_ >= other.p_;
        }

        target_ptr& operator+=(std::ptrdiff_t offset)
        {
            HPX_ASSERT(p_);
            p_ += offset;
            return *this;
        }

        target_ptr& operator-=(std::ptrdiff_t offset)
        {
            HPX_ASSERT(p_);
            p_ -= offset;
            return *this;
        }

        std::ptrdiff_t operator-(target_ptr const& other) const
        {
            return p_ - other.p_;
        }

        target_ptr operator-(std::ptrdiff_t offset) const
        {
            return target_ptr(p_ - offset, *tgt_);
        }

        target_ptr operator+(std::ptrdiff_t offset) const
        {
            return target_ptr(p_ + offset, *tgt_);
        }

#if defined(__CUDA_ARCH__)
        T const& operator*() const
        {
            return *p_;
        }

        T& operator*()
        {
            return *p_;
        }

        T const& operator[](std::ptrdiff_t offset) const
        {
            return *(p_ + offset);
        }

        T& operator[](std::ptrdiff_t offset)
        {
            return *(p_ + offset);
        }
#else
        value_proxy<T> operator*() const
        {
            return value_proxy<T>(p_, *tgt_);
        }

        value_proxy<T> operator[](std::ptrdiff_t offset)
        {
            return value_proxy<T>(p_ + offset, *tgt_);
        }
#endif

        HPX_HOST_DEVICE
        operator T*()
        {
            return p_;
        }

        T* device_ptr()
        {
            return p_;
        }

    private:
        T* p_;
        target* tgt_;
    };
}}}

#endif
