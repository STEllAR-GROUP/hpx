//  Copyright (c) 2007-2015 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_ACTIONS_ACTION_PRIORITY_HPP
#define HPX_ACTIONS_ACTION_PRIORITY_HPP

#include <hpx/lcos/async_fwd.hpp>
#include <hpx/runtime/threads/thread_enums.hpp>
#include <hpx/traits/action_priority.hpp>

namespace hpx { namespace actions
{
    template <typename Action>
    threads::thread_priority action_priority()
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type_;
        threads::thread_priority priority =
            static_cast<threads::thread_priority>(
                traits::action_priority<action_type_>::value);
        if (priority == threads::thread_priority_default)
            priority = threads::thread_priority_normal;
        return priority;
    }
}}

#endif
