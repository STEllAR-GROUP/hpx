//  Copyright (c) 2016 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/actions_base/traits/extract_action.hpp>

namespace hpx { namespace traits {

    ///////////////////////////////////////////////////////////////////////////
    // Trait to determine the continuation type for an action
    template <typename Action, typename Enable = void>
    struct action_continuation
    {
        using type = typename hpx::traits::extract_action<
            Action>::type::continuation_type;
    };
}}    // namespace hpx::traits
