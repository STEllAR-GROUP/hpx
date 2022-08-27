//  Copyright (c) 2007-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING)
#include <hpx/modules/serialization.hpp>

namespace hpx::traits {

    ///////////////////////////////////////////////////////////////////////////
    // Customization point for action serialization filter
    template <typename Action, typename Enable = void>
    struct action_serialization_filter
    {
        // return a new instance of a serialization filter
        static constexpr serialization::binary_filter* call() noexcept
        {
            return nullptr;    // by default actions don't have a serialization filter
        }
    };
}    // namespace hpx::traits

#endif
