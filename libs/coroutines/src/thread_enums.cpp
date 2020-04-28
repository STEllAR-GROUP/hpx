//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach, Katelyn Kufahl
//  Copyright (c) 2008-2009 Chirag Dekate, Anshul Tandon
//  Copyright (c) 2015 Patricia Grubel
//  Copyright (c) 2017 Shoshana Jakobovits
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/coroutines/thread_enums.hpp>

namespace hpx { namespace threads {
    ///////////////////////////////////////////////////////////////////////
    namespace strings {

        // clang-format off
        char const* const thread_state_names[] = {
            "unknown",
            "active",
            "pending",
            "suspended",
            "depleted",
            "terminated",
            "staged",
            "pending_do_not_schedule",
            "pending_boost"
        };
        // clang-format on

    }    // namespace strings

    char const* get_thread_state_name(thread_state_enum state)
    {
        if (state > pending_boost)
            return "unknown";
        return strings::thread_state_names[state];
    }

    char const* get_thread_state_name(thread_state state)
    {
        return get_thread_state_name(state.state());
    }

    ///////////////////////////////////////////////////////////////////////
    namespace strings {

        // clang-format off
        char const* const thread_state_ex_names[] = {
            "wait_unknown",
            "wait_signaled",
            "wait_timeout",
            "wait_terminate",
            "wait_abort"
        };
        // clang-format on

    }    // namespace strings

    char const* get_thread_state_ex_name(thread_state_ex_enum state_ex)
    {
        if (state_ex > wait_abort)
            return "wait_unknown";
        return strings::thread_state_ex_names[state_ex];
    }

    ///////////////////////////////////////////////////////////////////////
    namespace strings {

        // clang-format off
        char const* const thread_priority_names[] = {
            "default",
            "low",
            "normal",
            "high (recursive)",
            "boost",
            "high (non-recursive)",
        };
        // clang-format on
    }    // namespace strings

    char const* get_thread_priority_name(thread_priority priority)
    {
        if (priority < thread_priority_default ||
            priority > thread_priority_high)
        {
            return "unknown";
        }
        return strings::thread_priority_names[priority];
    }

    namespace strings {

        // clang-format off
        char const* const stack_size_names[] = {
            "small",
            "medium",
            "large",
            "huge",
            "nostack",
        };
        // clang-format on

    }    // namespace strings

    char const* get_stack_size_enum_name(thread_stacksize size)
    {
        if (size == thread_stacksize_unknown)
            return "unknown";

        if (size < thread_stacksize_small || size > thread_stacksize_nostack)
            return "custom";

        return strings::stack_size_names[size - 1];
    }
}}    // namespace hpx::threads
