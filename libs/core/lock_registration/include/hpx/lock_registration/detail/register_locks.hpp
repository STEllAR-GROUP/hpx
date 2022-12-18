//  Copyright (c) 2007-2022 Hartmut Kaiser
//  Copyright (c) 2014 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/concepts/has_member_xxx.hpp>
#include <hpx/functional/function.hpp>
#include <hpx/type_support/unused.hpp>

#include <cstddef>
#include <map>
#include <memory>
#include <type_traits>
#include <utility>

#ifdef HPX_HAVE_VERIFY_LOCKS_BACKTRACE
#include <string>
#endif

///////////////////////////////////////////////////////////////////////////////
namespace hpx::util {

    struct register_lock_data
    {
    };

    // Always provide function exports, which guarantees ABI compatibility of
    // Debug and Release builds.

#if defined(HPX_HAVE_VERIFY_LOCKS)

    namespace detail {

        struct HPX_CORE_EXPORT lock_data
        {
            explicit lock_data(std::size_t trace_depth);
            lock_data(register_lock_data* data, std::size_t trace_depth);

            ~lock_data();

            bool ignore_;
            register_lock_data* user_data_;
#ifdef HPX_HAVE_VERIFY_LOCKS_BACKTRACE
            std::string backtrace_;
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
        void const* lock, register_lock_data* data = nullptr) noexcept;
    HPX_CORE_EXPORT bool unregister_lock(void const* lock) noexcept;
    HPX_CORE_EXPORT void verify_no_locks();
    HPX_CORE_EXPORT void force_error_on_lock();
    HPX_CORE_EXPORT void enable_lock_detection() noexcept;
    HPX_CORE_EXPORT void disable_lock_detection() noexcept;
    HPX_CORE_EXPORT void trace_depth_lock_detection(std::size_t value) noexcept;
    HPX_CORE_EXPORT void ignore_lock(void const* lock) noexcept;
    HPX_CORE_EXPORT void reset_ignored(void const* lock) noexcept;
    HPX_CORE_EXPORT void ignore_all_locks() noexcept;
    HPX_CORE_EXPORT void reset_ignored_all() noexcept;

    using registered_locks_error_handler_type = hpx::function<void()>;

    /// Sets a handler which gets called when verifying that no locks are held
    /// fails. Can be used to print information at the point of failure such as
    /// a backtrace.
    HPX_CORE_EXPORT void set_registered_locks_error_handler(
        registered_locks_error_handler_type) noexcept;

    using register_locks_predicate_type = hpx::function<bool()>;

    /// Sets a predicate which gets called each time a lock is registered,
    /// unregistered, or when locks are verified. If the predicate returns
    /// false, the corresponding function will not register, unregister, or
    /// verify locks. If it returns true the corresponding function may
    /// register, unregister, or verify locks, depending on other factors (such
    /// as if lock detection is enabled globally). The predicate may return
    /// different values depending on context.
    HPX_CORE_EXPORT void set_register_locks_predicate(
        register_locks_predicate_type) noexcept;

    ///////////////////////////////////////////////////////////////////////////
    struct ignore_all_while_checking
    {
        ignore_all_while_checking() noexcept
        {
            ignore_all_locks();
        }

        ~ignore_all_while_checking() noexcept
        {
            reset_ignored_all();
        }
    };

    namespace detail {

        HPX_HAS_MEMBER_XXX_TRAIT_DEF(mutex)
    }

    template <typename Lock,
        typename Enable = std::enable_if_t<detail::has_mutex_v<Lock>>>
    struct ignore_while_checking
    {
        explicit ignore_while_checking(Lock const* lock) noexcept
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

    template <typename Lock, typename Enable = void>
    struct ignore_while_checking
    {
        explicit constexpr ignore_while_checking(Lock const* /*lock*/) noexcept
        {
        }
    };

    struct ignore_all_while_checking
    {
        constexpr ignore_all_while_checking() noexcept {}
    };

    constexpr inline bool register_lock(
        void const*, util::register_lock_data* = nullptr) noexcept
    {
        return true;
    }
    constexpr inline bool unregister_lock(void const*) noexcept
    {
        return true;
    }
    constexpr inline void verify_no_locks() noexcept {}
    constexpr inline void force_error_on_lock() noexcept {}
    constexpr inline void enable_lock_detection() noexcept {}
    constexpr inline void disable_lock_detection() noexcept {}
    constexpr inline void trace_depth_lock_detection(
        std::size_t /*value*/) noexcept
    {
    }
    constexpr inline void ignore_lock(void const* /*lock*/) noexcept {}
    constexpr inline void reset_ignored(void const* /*lock*/) noexcept {}

    constexpr inline void ignore_all_locks() noexcept {}
    constexpr inline void reset_ignored_all() noexcept {}

    struct held_locks_data
    {
    };

    constexpr inline held_locks_data* get_held_locks_data() noexcept
    {
        return nullptr;
    }

    constexpr inline void set_held_locks_data(held_locks_data*) noexcept {}

#endif
}    // namespace hpx::util
