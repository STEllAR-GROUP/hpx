//  Copyright (c) 2007-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/async_base/sync.hpp>

#include <type_traits>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx {

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {

        // dispatch point used for sync<Action> implementations
        template <typename Action, typename Func, typename Enable = void>
        struct sync_action_dispatch;
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename F, typename... Ts>
    HPX_FORCEINLINE auto sync(F&& f, Ts&&... ts)
        -> decltype(detail::sync_action_dispatch<Action, std::decay_t<F>>::call(
            HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...));
}    // namespace hpx
