///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <hpx/config.hpp>
#include <hpx/compute_local/detail/get_proxy_type.hpp>
#include <hpx/compute_local/traits/allocator_traits.hpp>
#include <hpx/modules/iterator_support.hpp>

#include <cstddef>
#include <iterator>

namespace hpx::compute::detail {

    template <typename T, typename Allocator>
    struct iterator
      : hpx::util::iterator_adaptor<iterator<T, Allocator>,
            typename traits::allocator_traits<Allocator>::pointer,
            typename traits::allocator_traits<Allocator>::value_type,
            std::random_access_iterator_tag,
            typename traits::allocator_traits<Allocator>::reference>
    {
        using base_type = hpx::util::iterator_adaptor<iterator<T, Allocator>,
            typename traits::allocator_traits<Allocator>::pointer,
            typename traits::allocator_traits<Allocator>::value_type,
            std::random_access_iterator_tag,
            typename traits::allocator_traits<Allocator>::reference>;

        using proxy_type = typename get_proxy_type<T>::type*;

        using const_reference =
            typename traits::allocator_traits<Allocator>::const_reference;
        using target_type =
            typename traits::allocator_traits<Allocator>::target_type;

        HPX_HOST_DEVICE iterator() noexcept
          : base_type(nullptr)
          , target_(nullptr)
        {
        }

        // FIXME: should be private
        HPX_HOST_DEVICE
        iterator(typename traits::allocator_traits<Allocator>::pointer p,
            std::size_t pos, target_type const& target) noexcept
          : base_type(p + pos)
          , target_(&target)
        {
        }

        HPX_HOST_DEVICE iterator(iterator const& other) noexcept
          : base_type(other)
          , target_(other.target_)
        {
        }

        // NOLINTNEXTLINE(bugprone-unhandled-self-assignment)
        HPX_HOST_DEVICE iterator& operator=(iterator const& other)
        {
            this->base_type::operator=(other);
            target_ = other.target_;

            return *this;
        }

        HPX_HOST_DEVICE target_type const& target() const noexcept
        {
            return *target_;
        }

    private:
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
}    // namespace hpx::compute::detail
