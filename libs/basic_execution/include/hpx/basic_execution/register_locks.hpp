//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2014 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_REGISTER_LOCKS_JUN_26_2012_1029AM)
#define HPX_UTIL_REGISTER_LOCKS_JUN_26_2012_1029AM

#include <hpx/config.hpp>
#include <hpx/concepts/has_member_xxx.hpp>

#include <functional>
#include <type_traits>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util {

    struct register_lock_data
    {
    };

    // Always provide function exports, which guarantees ABI compatibility of
    // Debug and Release builds.

    template <typename Lock, typename Enable = void>
    struct ignore_while_checking;

#if defined(HPX_HAVE_VERIFY_LOCKS) || defined(HPX_EXPORTS) ||                  \
    defined(HPX_MODULE_EXPORTS)

    ///////////////////////////////////////////////////////////////////////////
    HPX_API_EXPORT bool register_lock(
        void const* lock, register_lock_data* data = nullptr);
    HPX_API_EXPORT bool unregister_lock(void const* lock);
    HPX_API_EXPORT void verify_no_locks();
    HPX_API_EXPORT void force_error_on_lock();
    HPX_API_EXPORT void enable_lock_detection();
    HPX_API_EXPORT void ignore_lock(void const* lock);
    HPX_API_EXPORT void reset_ignored(void const* lock);
    HPX_API_EXPORT void ignore_all_locks();
    HPX_API_EXPORT void reset_ignored_all();

    using registered_locks_error_handler_type = std::function<void()>;

    /// Sets a handler which gets called when verifying that no locks are held
    /// fails. Can be used to print information at the point of failure such as
    /// a backtrace.
    HPX_API_EXPORT void set_registered_locks_error_handler(
        registered_locks_error_handler_type);

    using register_locks_predicate_type = std::function<bool()>;

    /// Sets a predicate which gets called each time a lock is registered,
    /// unregistered, or when locks are verified. If the predicate returns
    /// false, the corresponding function will not register, unregister, or
    /// verify locks. If it returns true the corresponding function may
    /// register, unregister, or verify locks, depending on other factors (such
    /// as if lock detection is enabled globally). The predicate may return
    /// different values depending on context.
    HPX_API_EXPORT void set_register_locks_predicate(
        register_locks_predicate_type);

    ///////////////////////////////////////////////////////////////////////////
    struct ignore_all_while_checking
    {
        ignore_all_while_checking()
        {
            ignore_all_locks();
        }

        ~ignore_all_while_checking()
        {
            reset_ignored_all();
        }
    };

    namespace detail {
        HPX_HAS_MEMBER_XXX_TRAIT_DEF(mutex)
    }

    template <typename Lock>
    struct ignore_while_checking<Lock,
        typename std::enable_if<detail::has_mutex<Lock>::value>::type>
    {
        ignore_while_checking(Lock const* lock)
          : mtx_(lock->mutex())
        {
            ignore_lock(mtx_);
        }

        ~ignore_while_checking()
        {
            reset_ignored(mtx_);
        }

        void const* mtx_;
    };

#else

    template <typename Lock, typename Enable>
    struct ignore_while_checking
    {
        ignore_while_checking(void const* /*lock*/) {}
    };

    struct ignore_all_while_checking
    {
        ignore_all_while_checking() {}
    };

    inline bool register_lock(void const*, util::register_lock_data* = nullptr)
    {
        return true;
    }
    inline bool unregister_lock(void const*)
    {
        return true;
    }
    inline void verify_no_locks() {}
    inline void force_error_on_lock() {}
    inline void enable_lock_detection() {}
    inline void ignore_lock(void const* /*lock*/) {}
    inline void reset_ignored(void const* /*lock*/) {}

    inline void ignore_all_locks() {}
    inline void reset_ignored_all() {}
#endif
}}    // namespace hpx::util

#endif
