//  Copyright (c) 2021-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/iterator_support/counting_iterator.hpp>
#include <hpx/iterator_support/iterator_range.hpp>

namespace hpx::util {

    ///////////////////////////////////////////////////////////////////////////
    template <typename Incrementable>
    struct counting_shape
      : hpx::util::iterator_range<hpx::util::counting_iterator<Incrementable>>
    {
        using base_type = hpx::util::iterator_range<
            hpx::util::counting_iterator<Incrementable>>;

        HPX_HOST_DEVICE explicit constexpr counting_shape(
            Incrementable n) noexcept
          : base_type(hpx::util::counting_iterator(Incrementable(0)),
                hpx::util::counting_iterator(n))
        {
        }

        HPX_HOST_DEVICE constexpr counting_shape(
            Incrementable b, Incrementable e) noexcept
          : base_type(hpx::util::counting_iterator(b),
                hpx::util::counting_iterator(e))
        {
        }
    };

    template <typename Incrementable>
    counting_shape(Incrementable) -> counting_shape<Incrementable>;

    template <typename Incrementable>
    counting_shape(Incrementable, Incrementable)
        -> counting_shape<Incrementable>;

    namespace detail {

        template <typename Incrementable>
        HPX_DEPRECATED_V(1, 9,
            "hpx::util::detail::make_counting_shape is deprecated, use "
            "hpx::util::counting_shape instead")
        HPX_HOST_DEVICE
            constexpr counting_shape<Incrementable> make_counting_shape(
                Incrementable n) noexcept
        {
            return counting_shape(n);
        }

        template <typename Incrementable>
        HPX_DEPRECATED_V(1, 9,
            "hpx::util::detail::make_counting_shape is deprecated, use "
            "hpx::util::counting_shape instead")
        HPX_HOST_DEVICE
            constexpr counting_shape<Incrementable> make_counting_shape(
                Incrementable b, Incrementable e) noexcept
        {
            return counting_shape(b, e);
        }
    }    // namespace detail
}    // namespace hpx::util
