//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/actions_base/basic_action_fwd.hpp>
#include <hpx/async_base/launch_policy.hpp>
#include <hpx/async_local/async_fwd.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/naming_base/id_type.hpp>

#include <type_traits>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx {
    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        // dispatch point used for async_cb implementations
        template <typename Func, typename Enable = void>
        struct async_cb_dispatch;

        // dispatch point used for async_cb<Action> implementations
        template <typename Action, typename Func, typename Enable = void>
        struct async_cb_action_dispatch;
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    // MSVC complains about ambiguities if it sees this forward declaration
    template <typename Action, typename F, typename... Ts>
    HPX_FORCEINLINE auto async_cb(F&& f, Ts&&... ts)
        -> decltype(detail::async_cb_action_dispatch<Action,
            typename std::decay<F>::type>::call(std::forward<F>(f),
            std::forward<Ts>(ts)...));
}    // namespace hpx
