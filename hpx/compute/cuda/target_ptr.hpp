///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#ifndef HPX_COMPUTE_CUDA_TARGET_PTR_HPP
#define HPX_COMPUTE_CUDA_TARGET_PTR_HPP

#include <hpx/config.hpp>

#if defined(HPX_HAVE_CUDA) && defined(__CUDACC__)

#include <hpx/util/assert.hpp>
#include <hpx/util/iterator_adaptor.hpp>

#include <hpx/compute/cuda/target.hpp>
#include <hpx/compute/cuda/value_proxy.hpp>
#include <hpx/compute/detail/get_proxy_type.hpp>

#include <cstddef>

namespace hpx { namespace compute { namespace cuda
{
    template <typename T>
    class target_ptr
      : public hpx::util::iterator_adaptor<
            target_ptr<T>, T*,
#if defined(__CUDA_ARCH__)
            T, std::random_access_iterator_tag, T&, std::ptrdiff_t, T*
#else
            value_proxy<T>, std::random_access_iterator_tag, value_proxy<T>,
            std::ptrdiff_t, T*
#endif
        >
    {
        typedef hpx::util::iterator_adaptor<
                target_ptr<T>, T*,
#if defined(__CUDA_ARCH__)
                T, std::random_access_iterator_tag, T&, std::ptrdiff_t, T*
#else
                value_proxy<T>, std::random_access_iterator_tag, value_proxy<T>,
                std::ptrdiff_t, T*
#endif
            > base_type;

    public:
        typedef
            typename compute::detail::get_proxy_type<T>::type *
            proxy_type;

        HPX_HOST_DEVICE target_ptr()
          : base_type(nullptr)
          , tgt_(nullptr)
        {}

        explicit HPX_HOST_DEVICE target_ptr(std::nullptr_t)
          : base_type(nullptr)
          , tgt_(nullptr)
        {}

        target_ptr(T* p, target & tgt)
          : base_type(p)
          , tgt_(&tgt)
        {}

        HPX_HOST_DEVICE
        target_ptr(target_ptr const& rhs)
          : base_type(rhs)
          , tgt_(rhs.tgt_)
        {}

        HPX_HOST_DEVICE
        target_ptr& operator=(target_ptr const& rhs)
        {
            this->base_type::operator=(rhs);
            tgt_ = rhs.tgt_;
            return *this;
        }

        HPX_HOST_DEVICE
        explicit operator bool() const
        {
            return this->base() != nullptr;
        }

        HPX_HOST_DEVICE
        friend bool operator==(target_ptr const& lhs, std::nullptr_t)
        {
            return lhs.base() == nullptr;
        }

        HPX_HOST_DEVICE
        friend bool operator!=(target_ptr const& lhs, std::nullptr_t)
        {
            return lhs.base() != nullptr;
        }

        HPX_HOST_DEVICE
        friend bool operator==(std::nullptr_t, target_ptr const& rhs)
        {
            return nullptr == rhs.base();
        }

        HPX_HOST_DEVICE
        friend bool operator!=(std::nullptr_t, target_ptr const& rhs)
        {
            return nullptr != rhs.base();
        }

    public:

#if defined(__CUDA_ARCH__)
        HPX_DEVICE operator T*() const
        {
            return this->base();
        }
#else
        // Note : need to define implicit cast at host_side because of invoke()
        //        which is defined host_device. This function should never be
        //        executed.
        HPX_HOST operator T*() const
        {
            HPX_ASSERT(false);
            return nullptr;
        }

    private:
        friend class hpx::util::iterator_core_access;

        typename base_type::reference dereference() const
        {
            return value_proxy<T>(this->base(), *tgt_);
        }
#endif

    public:
        T* device_ptr() const
        {
            return this->base();
        }

    private:
        target* tgt_;
    };
}}}

#endif
#endif
