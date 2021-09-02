//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/local/config.hpp>

#include <type_traits>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace detail {
    // dispatch point used for apply implementations
    template <typename Func, typename Enable = void>
    struct apply_dispatch;
}}    // namespace hpx::detail

namespace hpx {
    template <typename F, typename... Ts>
    HPX_FORCEINLINE bool apply(F&& f, Ts&&... ts)
    {
        return detail::apply_dispatch<typename std::decay<F>::type>::call(
            std::forward<F>(f), std::forward<Ts>(ts)...);
    }
}    // namespace hpx
