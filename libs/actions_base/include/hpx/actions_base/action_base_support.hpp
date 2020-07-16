//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//  Copyright (c)      2011 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file action_base_support.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/actions_base/actions_base_fwd.hpp>
#include <hpx/threading_base/thread_helpers.hpp>
#include <hpx/threading_base/thread_init_data.hpp>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
/// \namespace actions
namespace hpx { namespace actions { namespace detail {

    /// \cond NOINTERNAL
    ///////////////////////////////////////////////////////////////////////
    // Figure out what priority the action has to be be associated with
    // A dynamically specified default priority results in using the static
    // Priority.
    template <threads::thread_priority Priority>
    struct thread_priority
    {
        static threads::thread_priority call(threads::thread_priority priority)
        {
            if (priority == threads::thread_priority_default)
                return Priority;
            return priority;
        }
    };

    // If the static Priority is default, a dynamically specified default
    // priority results in using the normal priority.
    template <>
    struct thread_priority<threads::thread_priority_default>
    {
        static threads::thread_priority call(threads::thread_priority priority)
        {
            // The mapping to 'normal' is now done at the last possible
            // moment in the scheduler.
            //    if (priority == threads::thread_priority_default)
            //        return threads::thread_priority_normal;
            return priority;
        }
    };

    ///////////////////////////////////////////////////////////////////////
    // Figure out what stacksize the action has to be be associated with
    // A dynamically specified default stacksize results in using the static
    // Stacksize.
    template <threads::thread_stacksize Stacksize>
    struct thread_stacksize
    {
        static threads::thread_stacksize call(
            threads::thread_stacksize stacksize)
        {
            if (stacksize == threads::thread_stacksize_default)
                return Stacksize;
            return stacksize;
        }
    };

    // If the static Stacksize is default, a dynamically specified default
    // stacksize results in using the normal stacksize.
    template <>
    struct thread_stacksize<threads::thread_stacksize_default>
    {
        static threads::thread_stacksize call(
            threads::thread_stacksize stacksize)
        {
            if (stacksize == threads::thread_stacksize_default)
                return threads::thread_stacksize_minimal;
            return stacksize;
        }
    };
    /// \endcond
}}}    // namespace hpx::actions::detail

#include <hpx/config/warnings_suffix.hpp>
