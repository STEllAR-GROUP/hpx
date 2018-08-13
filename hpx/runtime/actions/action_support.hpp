//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//  Copyright (c)      2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file action_support.hpp

#if !defined(HPX_RUNTIME_ACTIONS_ACTION_SUPPORT_NOV_14_2008_0711PM)
#define HPX_RUNTIME_ACTIONS_ACTION_SUPPORT_NOV_14_2008_0711PM

#include <hpx/config.hpp>
#include <hpx/runtime/actions_fwd.hpp>
#include <hpx/runtime/components/pinned_ptr.hpp>
#include <hpx/runtime/parcelset_fwd.hpp>
#include <hpx/runtime/serialization/base_object.hpp>
#include <hpx/runtime/serialization/input_archive.hpp>
#include <hpx/runtime/serialization/output_archive.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/runtime/threads/thread_init_data.hpp>
#include <hpx/traits/action_remote_result.hpp>
#include <hpx/util/detail/pp/cat.hpp>
#include <hpx/util/detail/pp/nargs.hpp>
#include <hpx/util/tuple.hpp>
#if HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX)
#include <hpx/util/itt_notify.hpp>
#endif

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include <hpx/config/warnings_prefix.hpp>

/// \cond NOINTERNAL
namespace hpx { namespace traits
{
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        // If an action returns void, we need to do special things
        template <>
        struct action_remote_result_customization_point<void>
        {
            typedef util::unused_type type;
        };
    }
}}
/// \endcond

///////////////////////////////////////////////////////////////////////////////
/// \namespace actions
namespace hpx { namespace actions
{
    /// \cond NOINTERNAL

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename Action>
        char const* get_action_name()
#ifndef HPX_HAVE_AUTOMATIC_SERIALIZATION_REGISTRATION
        ;
#else
        {
            /// If you encounter this assert while compiling code, that means that
            /// you have a HPX_REGISTER_ACTION macro somewhere in a source file,
            /// but the header in which the action is defined misses a
            /// HPX_REGISTER_ACTION_DECLARATION
            static_assert(
                traits::needs_automatic_registration<Action>::value,
                "HPX_REGISTER_ACTION_DECLARATION missing");
            return debug::type_id<Action>::typeid_.type_id();
        }
#endif

        template <typename Action>
        std::uint32_t get_action_id()
        {
            static std::uint32_t id = get_action_id_from_name(
                get_action_name<Action>());
            return id;
        }

#if HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX)
        template <typename Action>
        util::itt::string_handle const& get_action_name_itt()
#ifndef HPX_HAVE_AUTOMATIC_SERIALIZATION_REGISTRATION
        ;
#else
        {
            static util::itt::string_handle sh = get_action_name<Action>();
            return sh;
        }
#endif
#endif
    }


    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        // Figure out what priority the action has to be be associated with
        // A dynamically specified default priority results in using the static
        // Priority.
        template <threads::thread_priority Priority>
        struct thread_priority
        {
            static threads::thread_priority
            call(threads::thread_priority priority)
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
            static threads::thread_priority
            call(threads::thread_priority priority)
            {
//              The mapping to 'normal' is now done at the last possible moment
//              in the scheduler.
//                 if (priority == threads::thread_priority_default)
//                     return threads::thread_priority_normal;
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
            static threads::thread_stacksize
            call(threads::thread_stacksize stacksize)
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
            static threads::thread_stacksize
            call(threads::thread_stacksize stacksize)
            {
                if (stacksize == threads::thread_stacksize_default)
                    return threads::thread_stacksize_minimal;
                return stacksize;
            }
        };
    }
    /// \endcond
}}

#include <hpx/config/warnings_suffix.hpp>

#endif
