//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/async_base/async.hpp>
#include <hpx/type_support/decay.hpp>

#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx {
    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        // dispatch point used for async<Action> implementations
        template <typename Action, typename Func, typename Enable = void>
        struct async_action_dispatch;
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename F, typename... Ts>
    HPX_FORCEINLINE auto async(F&& f, Ts&&... ts)
        -> decltype(detail::async_action_dispatch<Action,
            typename util::decay<F>::type>::call(std::forward<F>(f),
            std::forward<Ts>(ts)...));
}    // namespace hpx
