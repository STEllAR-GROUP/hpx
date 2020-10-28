//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/actions/continuation.hpp>
#include <hpx/actions/trigger.hpp>
#include <hpx/actions_base/traits/action_trigger_continuation_fwd.hpp>

#include <utility>

namespace hpx { namespace traits {

    ///////////////////////////////////////////////////////////////////////////
    // Trait to determine the continuation type for an action
    template <typename Result, typename RemoteResult>
    struct action_trigger_continuation<
        actions::typed_continuation<Result, RemoteResult>>
    {
        template <typename F, typename... Ts>
        HPX_FORCEINLINE static void call(
            actions::typed_continuation<Result, RemoteResult>&& cont, F&& f,
            Ts&&... ts) noexcept
        {
            actions::trigger(
                std::move(cont), std::forward<F>(f), std::forward<Ts>(ts)...);
        }
    };
}}    // namespace hpx::traits
