//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_CXX17_COPY_ELISION) &&                                    \
    defined(HPX_HAVE_CXX17_OPTIONAL_COPY_ELISION)

#include <functional>    // for std::hash<std::optional<T>>
#include <optional>

namespace hpx {

    using std::bad_optional_access;
    using std::make_optional;
    using std::nullopt;
    using std::nullopt_t;
    using std::optional;
}    // namespace hpx

#else

#include <hpx/datastructures/detail/optional.hpp>

namespace hpx {

    using hpx::optional_ns::bad_optional_access;
    using hpx::optional_ns::make_optional;
    using hpx::optional_ns::nullopt;
    using hpx::optional_ns::nullopt_t;
    using hpx::optional_ns::optional;
}    // namespace hpx

#endif

namespace hpx::util {

    using nullopt_t HPX_DEPRECATED_V(1, 8,
        "hpx::util::nullopt_t is deprecated. Please use hpx::nullopt_t "
        "instead.") = hpx::nullopt_t;

    HPX_DEPRECATED_V(1, 8,
        "hpx::util::nullopt is deprecated. Please use hpx::nullopt instead.")
    constexpr hpx::nullopt_t nullopt = hpx::nullopt;

    template <typename T>
    using optional HPX_DEPRECATED_V(1, 8,
        "hpx::util::optional is deprecated. Please use hpx::optional "
        "instead.") = hpx::optional<T>;

    using bad_optional_access HPX_DEPRECATED_V(1, 8,
        "hpx::util::bad_optional_access is deprecated. Please use "
        "hpx::bad_optional_access instead.") = hpx::bad_optional_access;

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    HPX_DEPRECATED_V(1, 8,
        "hpx::util::make_optional is deprecated. Please use hpx::optional "
        "instead.")
    constexpr auto make_optional(T&& t)
    {
        return hpx::make_optional(HPX_FORWARD(T, t));
    }

    template <typename T, typename... Ts>
    HPX_DEPRECATED_V(1, 8,
        "hpx::util::make_optional is deprecated. Please use hpx::optional "
        "instead.")
    constexpr auto make_optional(Ts&&... ts)
    {
        return hpx::make_optional(HPX_FORWARD(Ts, ts)...);
    }

    template <typename T, typename U, typename... Ts>
    HPX_DEPRECATED_V(1, 8,
        "hpx::util::make_optional is deprecated. Please use hpx::optional "
        "instead.")
    constexpr auto make_optional(std::initializer_list<U> il, Ts&&... ts)
    {
        return hpx::make_optional(il, HPX_FORWARD(Ts, ts)...);
    }
}    // namespace hpx::util
