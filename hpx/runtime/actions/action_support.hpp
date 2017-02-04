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
#include <hpx/lcos/future.hpp>
#include <hpx/runtime/actions_fwd.hpp>
#include <hpx/runtime/components/pinned_ptr.hpp>
#include <hpx/runtime/parcelset_fwd.hpp>
#include <hpx/runtime/serialization/base_object.hpp>
#include <hpx/runtime/serialization/input_archive.hpp>
#include <hpx/runtime/serialization/output_archive.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/runtime/threads/thread_init_data.hpp>
#include <hpx/traits/action_remote_result.hpp>
#include <hpx/traits/is_bitwise_serializable.hpp>
#include <hpx/util/detail/count_num_args.hpp>
#include <hpx/util/tuple.hpp>
#if HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX)
#include <hpx/util/itt_notify.hpp>
#endif

#include <boost/preprocessor/cat.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include <hpx/config/warnings_prefix.hpp>

/// \cond NOINTERNAL
namespace hpx { namespace actions { namespace detail
{
    struct action_serialization_data
    {
        action_serialization_data()
          : parent_locality_(naming::invalid_locality_id)
          , parent_id_(static_cast<std::uint64_t>(-1))
          , parent_phase_(0)
          , priority_(static_cast<threads::thread_priority>(0))
          , stacksize_(static_cast<threads::thread_stacksize>(0))
        {}

        action_serialization_data(std::uint32_t parent_locality,
                std::uint64_t parent_id,
                std::uint64_t parent_phase,
                threads::thread_priority priority,
                threads::thread_stacksize stacksize)
          : parent_locality_(parent_locality)
          , parent_id_(parent_id)
          , parent_phase_(parent_phase)
          , priority_(priority)
          , stacksize_(stacksize)
        {}

        std::uint32_t parent_locality_;
        std::uint64_t parent_id_;
        std::uint64_t parent_phase_;
        threads::thread_priority priority_;
        threads::thread_stacksize stacksize_;

        template <class Archive>
        void serialize(Archive& ar, unsigned)
        {
            ar & parent_id_ & parent_phase_ & parent_locality_
               & priority_ & stacksize_;
        }
    };
}}}

HPX_IS_BITWISE_SERIALIZABLE(hpx::actions::detail::action_serialization_data)

namespace hpx { namespace traits
{
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        // If an action returns a future, we need to do special things
        template <>
        struct action_remote_result_customization_point<void>
        {
            typedef util::unused_type type;
        };

        template <typename Result>
        struct action_remote_result_customization_point<lcos::future<Result> >
        {
            typedef Result type;
        };

        template <>
        struct action_remote_result_customization_point<lcos::future<void> >
        {
            typedef hpx::util::unused_type type;
        };

        template <typename Result>
        struct action_remote_result_customization_point<lcos::shared_future<Result> >
        {
            typedef Result type;
        };

        template <>
        struct action_remote_result_customization_point<lcos::shared_future<void> >
        {
            typedef hpx::util::unused_type type;
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
            return util::type_id<Action>::typeid_.type_id();
        }
#endif

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
                if (priority == threads::thread_priority_default)
                    return threads::thread_priority_normal;
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
