//  Copyright (c) 2016 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/actions/continuation_fwd.hpp>
#include <hpx/actions_base/traits/action_continuation_fwd.hpp>

namespace hpx { namespace traits {

    ///////////////////////////////////////////////////////////////////////////
    // Trait to determine the continuation type for an action
    template <typename Action, typename Enable>
    struct action_continuation
    {
        using type =
            hpx::actions::typed_continuation<typename Action::local_result_type,
                typename Action::remote_result_type>;
    };
}}    // namespace hpx::traits
