//  Copyright (c) 2007-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING)
#include <hpx/parcelset_base/parcelset_base_fwd.hpp>
#include <hpx/parcelset_base/policies/message_handler.hpp>

namespace hpx::traits {

    ///////////////////////////////////////////////////////////////////////////
    // Customization point for action message handlers
    template <typename Action, typename Enable = void>
    struct action_message_handler
    {
        // return a new instance of a serialization filter
        static constexpr parcelset::policies::message_handler* call(
            parcelset::locality const&) noexcept
        {
            return nullptr;    // by default actions don't have a message_handler
        }
    };
}    // namespace hpx::traits

#endif
