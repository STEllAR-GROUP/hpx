//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING)
#include <hpx/runtime/parcelset_fwd.hpp>

namespace hpx { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    // Customization point for action stack size
    template <typename Action, typename Enable = void>
    struct action_message_handler
    {
        // return a new instance of a serialization filter
        static parcelset::policies::message_handler* call(
            parcelset::parcelhandler*, parcelset::locality const&,
            parcelset::parcel const&)
        {
            return nullptr;   // by default actions don't have a message_handler
        }
    };
}}

#endif
