//  Copyright (c) 2007-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file sync.hpp

#pragma once

#include <hpx/config.hpp>

#include <type_traits>
#include <utility>

namespace hpx::detail {

    // dispatch point used for sync implementations
    template <typename Func, typename Enable = void>
    struct sync_dispatch;
}    // namespace hpx::detail

namespace hpx {

    /// The function template \a sync runs the function \a f synchronously and
    /// returns an \a hpx::future that will eventually hold the result of that
    /// function call.
    template <typename F, typename... Ts>
    HPX_FORCEINLINE decltype(auto) sync(F&& f, Ts&&... ts)
    {
        return detail::sync_dispatch<std::decay_t<F>>::call(
            HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
    }
}    // namespace hpx
