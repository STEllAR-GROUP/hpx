//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING)
#include <hpx/runtime/parcelset_fwd.hpp>
#include <hpx/serialization/serialization_fwd.hpp>

namespace hpx { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    // Customization point for action stack size
    template <typename Action, typename Enable = void>
    struct action_serialization_filter
    {
        // return a new instance of a serialization filter
        static serialization::binary_filter* call(parcelset::parcel const& /*p*/)
        {
            return nullptr;   // by default actions don't have a serialization filter
        }
    };
}}

#endif

