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
#include <type_traits>

namespace hpx::memory::detail {

    HPX_CXX_EXPORT template <typename Y, typename T>
    struct sp_convertible : std::is_convertible<Y*, T*>
    {
    };

    HPX_CXX_EXPORT template <typename Y, typename T>
    struct sp_convertible<Y, T[]>
    {
        static constexpr bool value = false;
    };

    HPX_CXX_EXPORT template <typename Y, typename T>
    struct sp_convertible<Y[], T[]>
    {
        static constexpr bool value = sp_convertible<Y[1], T[1]>::value;
    };

    HPX_CXX_EXPORT template <typename Y, std::size_t N, typename T>
    struct sp_convertible<Y[N], T[]>
    {
        static constexpr bool value = sp_convertible<Y[1], T[1]>::value;
    };

    HPX_CXX_EXPORT template <typename Y, typename T>
    inline constexpr bool sp_convertible_v = sp_convertible<Y, T>::value;
}    // namespace hpx::memory::detail
