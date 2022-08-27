//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace detail {
    // dispatch point used for async implementations
    template <typename Func, typename Enable = void>
    struct async_dispatch;
}}    // namespace hpx::detail

namespace hpx {
    template <typename F, typename... Ts>
    HPX_FORCEINLINE decltype(auto) async(F&& f, Ts&&... ts)
    {
        return detail::async_dispatch<typename std::decay<F>::type>::call(
            HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
    }
}    // namespace hpx
