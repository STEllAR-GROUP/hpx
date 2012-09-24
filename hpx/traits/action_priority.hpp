//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_ACTION_PRIORITY_SEP_03_2012_1138AM)
#define HPX_TRAITS_ACTION_PRIORITY_SEP_03_2012_1138AM

#include <hpx/hpx_fwd.hpp>

namespace hpx { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    // Customization point for action stack size
    template <typename Action, typename Enable>
    struct action_priority
    {
        enum { value = threads::thread_priority_default };
    };

    template <typename Action>
    struct action_priority<Action, typename Action::type>
      : action_priority<typename Action::type>
    {};
}}

#endif

