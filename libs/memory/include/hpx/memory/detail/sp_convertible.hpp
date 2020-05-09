//  detail/sp_convertible.hpp
//
//  Copyright 2008 Peter Dimov
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt

#pragma once

#include <hpx/config.hpp>

#include <cstddef>

namespace hpx { namespace memory { namespace detail {

    template <typename Y, typename T>
    struct sp_convertible
    {
        typedef char (&yes)[1];
        typedef char (&no)[2];

        static yes f(T*);
        static no f(...);

        HPX_STATIC_CONSTEXPR bool value =
            sizeof((f)(static_cast<Y*>(nullptr))) == sizeof(yes);
    };

    template <typename Y, typename T>
    struct sp_convertible<Y, T[]>
    {
        HPX_STATIC_CONSTEXPR bool value = false;
    };

    template <typename Y, typename T>
    struct sp_convertible<Y[], T[]>
    {
        HPX_STATIC_CONSTEXPR bool value = sp_convertible<Y[1], T[1]>::value;
    };

    template <typename Y, std::size_t N, typename T>
    struct sp_convertible<Y[N], T[]>
    {
        HPX_STATIC_CONSTEXPR bool value = sp_convertible<Y[1], T[1]>::value;
    };
}}}    // namespace hpx::memory::detail
