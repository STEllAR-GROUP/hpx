///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <hpx/config.hpp>

#include <type_traits>

namespace hpx::traits {

    HPX_CXX_CORE_EXPORT template <typename T>
    struct is_value_proxy : std::false_type
    {
    };

    HPX_CXX_CORE_EXPORT template <typename T>
    inline constexpr bool is_value_proxy_v = is_value_proxy<T>::value;

    HPX_CXX_CORE_EXPORT template <typename T>
    struct proxy_value
    {
        using type = T;
    };

    HPX_CXX_CORE_EXPORT template <typename T>
    using proxy_value_t = typename proxy_value<T>::type;
}    // namespace hpx::traits
