//  Copyright (c) 2007-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/coroutines/thread_enums.hpp>

namespace hpx { namespace traits {

    ///////////////////////////////////////////////////////////////////////////
    // Customization point for action stack size
    template <typename Action, typename Enable = void>
    struct action_stacksize
    {
        static constexpr threads::thread_stacksize value =
            threads::thread_stacksize::default_;
    };

    template <typename Action>
    inline constexpr threads::thread_stacksize action_stacksize_v =
        action_stacksize<Action>::value;
}}    // namespace hpx::traits
