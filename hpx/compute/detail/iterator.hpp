///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#ifndef HPX_COMPUTE_DETAIL_ITERATOR_HPP
#define HPX_COMPUTE_DETAIL_ITERATOR_HPP

#include <hpx/config.hpp>
#include <hpx/util/assert.hpp>

#include <hpx/compute/detail/get_proxy_type.hpp>
#include <hpx/compute/traits/allocator_traits.hpp>

#include <iterator>

namespace hpx { namespace compute { namespace detail
{
    template <typename T, typename Allocator>
    struct iterator
      : std::iterator<
            std::random_access_iterator_tag,
            T,
            std::ptrdiff_t,
            typename traits::allocator_traits<Allocator>::pointer,
            typename traits::allocator_traits<Allocator>::reference
        >
    {
        typedef
            typename get_proxy_type<T>::type *
            proxy_type;

        typedef
            typename traits::allocator_traits<Allocator>::value_type
            value_type;
        typedef
            typename traits::allocator_traits<Allocator>::pointer
            pointer;
        typedef
            typename traits::allocator_traits<Allocator>::reference
            reference;

        typedef std::random_access_iterator_tag iterator_category;
        typedef std::ptrdiff_t difference_type;

        typedef
            typename traits::allocator_traits<Allocator>::const_reference
            const_reference;
        typedef
            typename traits::allocator_traits<Allocator>::target_type
            target_type;

        HPX_HOST_DEVICE iterator()
          : p_(nullptr)
          , target_(nullptr)
        {}

        // FIXME: should be private
        HPX_HOST_DEVICE
        iterator(pointer p, std::size_t pos, target_type const& target)
          : p_(p + pos)
          , target_(&target)
        {}

        HPX_HOST_DEVICE iterator(iterator const& other)
          : p_(other.p_)
          , target_(other.target_)
        {}

        HPX_HOST_DEVICE iterator& operator=(iterator const& other)
        {
            p_ = other.p_;
            target_ = other.target_;

            return *this;
        }

        HPX_HOST_DEVICE iterator const& operator++()
        {
            HPX_ASSERT(p_);
            ++p_;
            return *this;
        }

        HPX_HOST_DEVICE iterator const& operator--()
        {
            HPX_ASSERT(p_);
            --p_;
            return *this;
        }

        HPX_HOST_DEVICE iterator operator++(int)
        {
            iterator tmp(*this);
            HPX_ASSERT(p_);
            ++p_;
            return tmp;
        }

        HPX_HOST_DEVICE iterator operator--(int)
        {
            iterator tmp(*this);
            HPX_ASSERT(p_);
            --p_;
            return tmp;
        }

        HPX_HOST_DEVICE bool operator==(iterator const& other) const
        {
            return p_ == other.p_;
        }

        HPX_HOST_DEVICE bool operator!=(iterator const& other) const
        {
            return p_ != other.p_;
        }

        HPX_HOST_DEVICE bool operator<(iterator const& other) const
        {
            return p_ < other.p_;
        }

        HPX_HOST_DEVICE bool operator>(iterator const& other) const
        {
            return p_ > other.p_;
        }

        HPX_HOST_DEVICE bool operator<=(iterator const& other) const
        {
            return p_ <= other.p_;
        }

        HPX_HOST_DEVICE bool operator>=(iterator const& other) const
        {
            return p_ >= other.p_;
        }

        HPX_HOST_DEVICE iterator& operator+=(std::ptrdiff_t offset)
        {
            HPX_ASSERT(p_);
            p_ += offset;
            return *this;
        }

        HPX_HOST_DEVICE iterator& operator-=(std::ptrdiff_t offset)
        {
            HPX_ASSERT(p_);
            p_ -= offset;
            return *this;
        }

        HPX_HOST_DEVICE iterator operator+(std::ptrdiff_t offset) const
        {
            iterator tmp(*this);
            tmp += offset;
            return tmp;
        }

        HPX_HOST_DEVICE iterator operator-(std::ptrdiff_t offset) const
        {
            iterator tmp(*this);
            tmp -= offset;
            return tmp;
        }

        HPX_HOST_DEVICE std::ptrdiff_t operator-(iterator const& other) const
        {
            return p_ - other.p_;
        }

        HPX_HOST_DEVICE reference operator*() const
        {
            HPX_ASSERT(p_);
            return *p_;
        }

        HPX_HOST_DEVICE pointer operator->() const
        {
            HPX_ASSERT(p_);
            return p_;
        }

        HPX_HOST_DEVICE reference operator[](std::size_t pos) const
        {
            HPX_ASSERT(p_);
            return *(p_ + pos);
        }

        HPX_HOST_DEVICE target_type const& target() const
        {
            return *target_;
        }

    private:
        pointer p_;
        target_type const* target_;
    };

    template <typename T, typename Allocator>
    struct reverse_iterator
    {
    };

    template <typename T, typename Allocator>
    struct const_reverse_iterator
    {
    };
}}}

#endif
