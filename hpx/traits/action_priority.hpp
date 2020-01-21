//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_TRAITS_ACTION_PRIORITY_SEP_03_2012_1138AM
#define HPX_TRAITS_ACTION_PRIORITY_SEP_03_2012_1138AM

#include <hpx/coroutines/thread_enums.hpp>

namespace hpx { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    // Customization point for action stack size
    template <typename Action, typename Enable = void>
    struct action_priority
    {
        enum { value = threads::thread_priority_default };
    };
}}

#endif

