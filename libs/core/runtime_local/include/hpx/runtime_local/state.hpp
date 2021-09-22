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

namespace hpx { namespace threads {

    // return whether thread manager is in the state described by 'mask'
    HPX_CORE_EXPORT bool threadmanager_is(state st);
    HPX_CORE_EXPORT bool threadmanager_is_at_least(state st);
}}    // namespace hpx::threads
