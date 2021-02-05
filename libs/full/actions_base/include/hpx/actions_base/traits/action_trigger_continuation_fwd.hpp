//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

namespace hpx { namespace traits {

    ///////////////////////////////////////////////////////////////////////////
    // Trait to determine the continuation type for an action
    template <typename Continuation, typename Enable = void>
    struct action_trigger_continuation
    {
        template <typename F, typename... Ts>
        static decltype(auto) call(Continuation&&, F&&, Ts&&...) noexcept;
    };
}}    // namespace hpx::traits
