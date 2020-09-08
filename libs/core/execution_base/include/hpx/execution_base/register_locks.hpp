//  Copyright (c) 2007-2020 Hartmut Kaiser
//  Copyright (c) 2014 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/concepts/has_member_xxx.hpp>
#include <hpx/modules/functional.hpp>

#include <cstddef>
#include <map>
#include <memory>
#ifdef HPX_HAVE_VERIFY_LOCKS_BACKTRACE
#include <string>
#endif
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

#if defined(HPX_HAVE_VERIFY_LOCKS) || defined(HPX_CORE_EXPORTS)

    namespace detail {

        struct HPX_CORE_EXPORT lock_data
        {
#ifdef HPX_HAVE_VERIFY_LOCKS
            lock_data(std::size_t trace_depth);
            lock_data(register_lock_data* data, std::size_t trace_depth);

            ~lock_data();

            bool ignore_;
            register_lock_data* user_data_;
#ifdef HPX_HAVE_VERIFY_LOCKS_BACKTRACE
            std::string backtrace_;
#endif
#endif
        };
    }    // namespace detail

    struct held_locks_data
    {
        using held_locks_map = std::map<void const*, detail::lock_data>;

        held_locks_data()
          : enabled_(true)
          , ignore_all_locks_(false)
        {
        }

        held_locks_map map_;
        bool enabled_;
        bool ignore_all_locks_;
    };

    ///////////////////////////////////////////////////////////////////////////
    HPX_CORE_EXPORT bool register_lock(
        void const* lock, register_lock_data* data = nullptr);
    HPX_CORE_EXPORT bool unregister_lock(void const* lock);
    HPX_CORE_EXPORT void verify_no_locks();
    HPX_CORE_EXPORT void force_error_on_lock();
    HPX_CORE_EXPORT void enable_lock_detection();
    HPX_CORE_EXPORT void disable_lock_detection();
    HPX_CORE_EXPORT void trace_depth_lock_detection(std::size_t value);
    HPX_CORE_EXPORT void ignore_lock(void const* lock);
    HPX_CORE_EXPORT void reset_ignored(void const* lock);
    HPX_CORE_EXPORT void ignore_all_locks();
    HPX_CORE_EXPORT void reset_ignored_all();

    using registered_locks_error_handler_type = util::function_nonser<void()>;

    /// Sets a handler which gets called when verifying that no locks are held
    /// fails. Can be used to print information at the point of failure such as
    /// a backtrace.
    HPX_CORE_EXPORT void set_registered_locks_error_handler(
        registered_locks_error_handler_type);

    using register_locks_predicate_type = util::function_nonser<bool()>;

    /// Sets a predicate which gets called each time a lock is registered,
    /// unregistered, or when locks are verified. If the predicate returns
    /// false, the corresponding function will not register, unregister, or
    /// verify locks. If it returns true the corresponding function may
    /// register, unregister, or verify locks, depending on other factors (such
    /// as if lock detection is enabled globally). The predicate may return
    /// different values depending on context.
    HPX_CORE_EXPORT void set_register_locks_predicate(
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

    // The following functions are used to store the held locks information
    // during thread suspension. The data is stored on a thread_local basis,
    // so we must make sure that locks the are being ignored are restored
    // after suspension even if the thread is being resumed on a different core.

    // retrieve the current thread_local data about held locks
    HPX_CORE_EXPORT std::unique_ptr<held_locks_data> get_held_locks_data();

    // set the current thread_local data about held locks
    HPX_CORE_EXPORT void set_held_locks_data(
        std::unique_ptr<held_locks_data>&& data);

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

    constexpr inline bool register_lock(
        void const*, util::register_lock_data* = nullptr)
    {
        return true;
    }
    constexpr inline bool unregister_lock(void const*)
    {
        return true;
    }
    constexpr inline void verify_no_locks() {}
    constexpr inline void force_error_on_lock() {}
    constexpr inline void enable_lock_detection() {}
    constexpr inline void disable_lock_detection() {}
    constexpr inline void trace_depth_lock_detection(std::size_t /*value*/) {}
    constexpr inline void ignore_lock(void const* /*lock*/) {}
    constexpr inline void reset_ignored(void const* /*lock*/) {}

    constexpr inline void ignore_all_locks() {}
    constexpr inline void reset_ignored_all() {}

    struct held_locks_data
    {
    };

    inline std::unique_ptr<held_locks_data> get_held_locks_data()
    {
        return std::unique_ptr<held_locks_data>();
    }

    constexpr inline void set_held_locks_data(
        std::unique_ptr<held_locks_data>&& /*data*/)
    {
    }

#endif
}}    // namespace hpx::util
