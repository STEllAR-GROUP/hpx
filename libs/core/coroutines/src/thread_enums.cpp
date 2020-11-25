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

#include <cstddef>

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

    char const* get_thread_state_name(thread_schedule_state state)
    {
        if (state > thread_schedule_state::pending_boost)
            return "unknown";
        return strings::thread_state_names[static_cast<std::size_t>(state)];
    }

    char const* get_thread_state_name(thread_state state)
    {
        return get_thread_state_name(state.state());
    }

    std::ostream& operator<<(std::ostream& os, thread_schedule_state const t)
    {
        os << get_thread_state_name(t) << " (" << static_cast<std::size_t>(t)
           << ")";
        return os;
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

    char const* get_thread_state_ex_name(thread_restart_state state_ex)
    {
        if (state_ex > thread_restart_state::abort)
            return "wait_unknown";
        return strings::thread_state_ex_names[static_cast<std::size_t>(
            state_ex)];
    }

    std::ostream& operator<<(std::ostream& os, thread_restart_state const t)
    {
        os << get_thread_state_ex_name(t) << " (" << static_cast<std::size_t>(t)
           << ")";
        return os;
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
        if (priority < thread_priority::default_ ||
            priority > thread_priority::high)
        {
            return "unknown";
        }
        return strings::thread_priority_names[static_cast<std::size_t>(
            priority)];
    }

    std::ostream& operator<<(std::ostream& os, thread_priority const t)
    {
        os << get_thread_priority_name(t) << " (" << static_cast<std::size_t>(t)
           << ")";
        return os;
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
        if (size == thread_stacksize::unknown)
            return "unknown";

        if (size < thread_stacksize::small_ || size > thread_stacksize::nostack)
            return "custom";

        return strings::stack_size_names[static_cast<std::size_t>(size) - 1];
    }

    std::ostream& operator<<(std::ostream& os, thread_stacksize const t)
    {
        os << get_stack_size_enum_name(t) << " (" << static_cast<std::size_t>(t)
           << ")";
        return os;
    }
}}    // namespace hpx::threads
