//  Copyright (c) 2007-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <type_traits>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx::detail {

    // dispatch point used for post implementations
    template <typename Func, typename Enable = void>
    struct post_dispatch;
}    // namespace hpx::detail

namespace hpx {

    template <typename F, typename... Ts>
    HPX_FORCEINLINE bool post(F&& f, Ts&&... ts)
    {
        return detail::post_dispatch<std::decay_t<F>>::call(
            HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
    }

    template <typename F, typename... Ts>
    HPX_DEPRECATED_V(1, 9, "hpx::apply is deprecated, use hpx::post instead")
    HPX_FORCEINLINE bool apply(F&& f, Ts&&... ts)
    {
        return detail::post_dispatch<std::decay_t<F>>::call(
            HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
    }
}    // namespace hpx
