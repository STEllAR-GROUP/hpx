////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <hpx/config.hpp>
#include <hpx/runtime_local/state.hpp>
#include <hpx/threading_base/scheduler_state.hpp>

namespace hpx
{
    namespace agas
    {
        // return whether resolver client is in state described by 'mask'
        HPX_EXPORT bool router_is(state st);
    }
}


