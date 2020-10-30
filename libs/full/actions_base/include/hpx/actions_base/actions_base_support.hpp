//  Copyright (c) 2007-2020 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//  Copyright (c)      2011 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file actions_base_support.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/actions_base/actions_base_fwd.hpp>
#include <hpx/actions_base/actions_base_support.hpp>
#include <hpx/actions_base/traits/action_remote_result.hpp>
#include <hpx/debugging/demangle_helper.hpp>
#include <hpx/serialization/traits/needs_automatic_registration.hpp>
#include <hpx/threading_base/thread_helpers.hpp>
#include <hpx/threading_base/thread_init_data.hpp>
#if HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX)
#include <hpx/modules/itt_notify.hpp>
#endif

#include <cstdint>

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
            if (priority == threads::thread_priority::default_)
                return Priority;
            return priority;
        }
    };

    // If the static Priority is default, a dynamically specified default
    // priority results in using the normal priority.
    template <>
    struct thread_priority<threads::thread_priority::default_>
    {
        static threads::thread_priority call(threads::thread_priority priority)
        {
            // The mapping to 'normal' is now done at the last possible
            // moment in the scheduler.
            //    if (priority == threads::thread_priority::default_)
            //        return threads::thread_priority::normal;
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
            if (stacksize == threads::thread_stacksize::default_)
                return Stacksize;
            return stacksize;
        }
    };

    // If the static Stacksize is default, a dynamically specified default
    // stacksize results in using the normal stacksize.
    template <>
    struct thread_stacksize<threads::thread_stacksize::default_>
    {
        static threads::thread_stacksize call(
            threads::thread_stacksize stacksize)
        {
            if (stacksize == threads::thread_stacksize::default_)
                return threads::thread_stacksize::minimal;
            return stacksize;
        }
    };

    ///////////////////////////////////////////////////////////////////////////
#if defined(HPX_HAVE_NETWORKING)
    template <typename Action>
    char const* get_action_name()
#if !defined(HPX_HAVE_AUTOMATIC_SERIALIZATION_REGISTRATION)
        ;
#else
    {
        /// If you encounter this assert while compiling code, that means that
        /// you have a HPX_REGISTER_ACTION macro somewhere in a source file,
        /// but the header in which the action is defined misses a
        /// HPX_REGISTER_ACTION_DECLARATION
        static_assert(traits::needs_automatic_registration<Action>::value,
            "HPX_REGISTER_ACTION_DECLARATION missing");
        return util::debug::type_id<Action>::typeid_.type_id();
    }
#endif
#else    // HPX_HAVE_NETWORKING
    template <typename Action>
    char const* get_action_name()
    {
        return util::debug::type_id<Action>::typeid_.type_id();
    }
#endif

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action>
    std::uint32_t get_action_id()
    {
        static std::uint32_t id =
            get_action_id_from_name(get_action_name<Action>());
        return id;
    }

#if HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX)
    template <typename Action>
    util::itt::string_handle const& get_action_name_itt()
#if !defined(HPX_HAVE_AUTOMATIC_SERIALIZATION_REGISTRATION)
        ;
#else
    {
        static util::itt::string_handle sh = get_action_name<Action>();
        return sh;
    }
#endif
#endif

    /// \endcond
}}}    // namespace hpx::actions::detail

#include <hpx/config/warnings_suffix.hpp>
