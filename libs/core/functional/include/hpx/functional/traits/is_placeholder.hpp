//  Copyright (c) 2013-2015 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// make inspect happy: hpxinspect:nodeprecatedname:boost::is_placeholder

#pragma once

#include <hpx/config.hpp>

#include <boost/bind/arg.hpp>

#include <functional>
#include <type_traits>

namespace hpx {
#if defined(DOXYGEN)
    /// If \p T is a standard, Boost, or HPX placeholder (_1, _2, _3, ...) then
    /// this template is derived from ``std::integral_constant<int, 1>``,
    /// ``std::integral_constant<int, 2>``, ``std::integral_constant<int, 3>``,
    /// respectively. Otherwise it is derived from ,
    /// ``std::integral_constant<int, 0>``.
    template <typename T>
    struct is_placeholder;
#else
    template <typename T>
    struct is_placeholder
      : std::integral_constant<int,
            std::is_placeholder_v<T> != 0 ? std::is_placeholder_v<T> :
                                            boost::is_placeholder<T>::value>
    {
    };

    template <typename T>
    struct is_placeholder<T const> : is_placeholder<T>
    {
    };

    template <typename T>
    inline constexpr int is_placeholder_v = is_placeholder<T>::value;
#endif
}    // namespace hpx

namespace hpx::traits {

    template <typename T>
    using is_placeholder HPX_DEPRECATED_V(1, 8,
        "hpx::traits::is_placeholder is deprecated, use "
        "hpx::is_placeholder instead") = hpx::is_placeholder<T>;

    template <typename T>
    HPX_DEPRECATED_V(1, 8,
        "hpx::traits::is_placeholder_v is deprecated, use "
        "hpx::is_placeholder_v instead")
    inline constexpr bool is_placeholder_v = hpx::is_placeholder_v<T>;
}    // namespace hpx::traits
