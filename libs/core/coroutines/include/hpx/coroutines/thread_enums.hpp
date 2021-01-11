//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx/coroutines/thread_enums.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/coroutines/detail/combined_tagged_state.hpp>

#include <cstddef>
#include <cstdint>
#include <iostream>

namespace hpx { namespace threads {

    ///////////////////////////////////////////////////////////////////////////
    // clang-format off

    /// \enum thread_schedule_state
    ///
    /// The \a thread_schedule_state enumerator encodes the current state of a
    /// \a thread instance
    enum class thread_schedule_state
    {
        unknown = 0,
        active = 1,                   /*!< thread is currently active (running,
                                           has resources) */
        pending = 2,                  /*!< thread is pending (ready to run, but
                                           no hardware resource available) */
        suspended = 3,                /*!< thread has been suspended (waiting
                                           for synchronization event, but still
                                           known and under control of the
                                           thread-manager) */
        depleted = 4,                 /*!< thread has been depleted (deeply
                                           suspended, it is not known to the
                                           thread-manager) */
        terminated = 5,               /*!< thread has been stopped an may be
                                           garbage collected */
        staged = 6,                   /*!< this is not a real thread state, but
                                           allows to reference staged task
                                           descriptions, which eventually will
                                           be converted into thread objects */
        pending_do_not_schedule = 7,  /*< this is not a real thread state,
                                          but allows to create a thread in
                                          pending state without scheduling it
                                          (internal, do not use) */
        pending_boost = 8             /*< this is not a real thread state,
                                          but allows to suspend a thread in
                                          pending state without high priority
                                          rescheduling */
    };
    // clang-format on

#define HPX_THREAD_STATE_UNSCOPED_ENUM_DEPRECATION_MSG                         \
    "The unscoped thread_state_enum names are deprecated. Please use "         \
    "thread_schedule_state::state instead."

    HPX_DEPRECATED_V(1, 6, HPX_THREAD_STATE_UNSCOPED_ENUM_DEPRECATION_MSG)
    static constexpr thread_schedule_state unknown =
        thread_schedule_state::unknown;
    HPX_DEPRECATED_V(1, 6, HPX_THREAD_STATE_UNSCOPED_ENUM_DEPRECATION_MSG)
    static constexpr thread_schedule_state active =
        thread_schedule_state::active;
    HPX_DEPRECATED_V(1, 6, HPX_THREAD_STATE_UNSCOPED_ENUM_DEPRECATION_MSG)
    static constexpr thread_schedule_state pending =
        thread_schedule_state::pending;
    HPX_DEPRECATED_V(1, 6, HPX_THREAD_STATE_UNSCOPED_ENUM_DEPRECATION_MSG)
    static constexpr thread_schedule_state suspended =
        thread_schedule_state::suspended;
    HPX_DEPRECATED_V(1, 6, HPX_THREAD_STATE_UNSCOPED_ENUM_DEPRECATION_MSG)
    static constexpr thread_schedule_state depleted =
        thread_schedule_state::depleted;
    HPX_DEPRECATED_V(1, 6, HPX_THREAD_STATE_UNSCOPED_ENUM_DEPRECATION_MSG)
    static constexpr thread_schedule_state terminated =
        thread_schedule_state::terminated;
    HPX_DEPRECATED_V(1, 6, HPX_THREAD_STATE_UNSCOPED_ENUM_DEPRECATION_MSG)
    static constexpr thread_schedule_state staged =
        thread_schedule_state::staged;
    HPX_DEPRECATED_V(1, 6, HPX_THREAD_STATE_UNSCOPED_ENUM_DEPRECATION_MSG)
    static constexpr thread_schedule_state pending_do_not_schedule =
        thread_schedule_state::pending_do_not_schedule;
    HPX_DEPRECATED_V(1, 6, HPX_THREAD_STATE_UNSCOPED_ENUM_DEPRECATION_MSG)
    static constexpr thread_schedule_state pending_boost =
        thread_schedule_state::pending_boost;
#undef HPX_THREAD_STATE_UNSCOPED_ENUM_DEPRECATION_MSG

    HPX_CORE_EXPORT std::ostream& operator<<(
        std::ostream& os, thread_schedule_state const t);

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Returns the name of the given state
    ///
    /// Get the readable string representing the name of the given
    /// thread_state constant.
    ///
    /// \param state this represents the thread state.
    HPX_CORE_EXPORT char const* get_thread_state_name(
        thread_schedule_state state);

    ///////////////////////////////////////////////////////////////////////////
    // clang-format off

    /// This enumeration lists all possible thread-priorities for HPX threads.
    ///
    enum class thread_priority
    {
        unknown = -1,
        default_ = 0, /*!< Will assign the priority of the
            task to the default (normal) priority. */
        low = 1,     /*!< Task goes onto a special low
            priority queue and will not be executed until all high/normal
            priority tasks are done, even if they are added after the low
            priority task. */
        normal = 2,  /*!< Task will be executed when it is
            taken from the normal priority queue, this is usually a first
            in-first-out ordering of tasks (depending on scheduler choice).
            This is the default priority. */
        high_recursive = 3, /*!< The task is a high priority
            task and any child tasks spawned by this task will be made high
            priority as well - unless they are specifically flagged as non
            default priority. */
        boost = 4, /*!< Same as \a thread_priority_high
            except that the thread will fall back to \a thread_priority_normal
            if resumed after being suspended. */
        high = 5,  /*!< Task goes onto a special high
            priority queue and will be executed before normal/low priority
            tasks are taken (some schedulers modify the behavior slightly and
            the documentation for those should be consulted). */
        bound = 6,          /*!< Task goes onto a special
            high priority queue and will never be stolen by another thread
            after initial assignment. This should be used for thread placement
            tasks such as OpenMP type for loops. */

        /// \cond NOINTERNAL
        // obsolete, kept for compatibility only
        critical = high_recursive,
        /// \endcond
    };
    // clang-format on

#define HPX_THREAD_PRIORITY_UNSCOPED_ENUM_DEPRECATION_MSG                      \
    "The unscoped thread_priority names are deprecated. Please use "           \
    "thread_priority::priority instead."

    HPX_DEPRECATED_V(1, 6, HPX_THREAD_PRIORITY_UNSCOPED_ENUM_DEPRECATION_MSG)
    static constexpr thread_priority thread_priority_unknown =
        thread_priority::unknown;
    HPX_DEPRECATED_V(1, 6, HPX_THREAD_PRIORITY_UNSCOPED_ENUM_DEPRECATION_MSG)
    static constexpr thread_priority thread_priority_default =
        thread_priority::default_;
    HPX_DEPRECATED_V(1, 6, HPX_THREAD_PRIORITY_UNSCOPED_ENUM_DEPRECATION_MSG)
    static constexpr thread_priority thread_priority_low = thread_priority::low;
    HPX_DEPRECATED_V(1, 6, HPX_THREAD_PRIORITY_UNSCOPED_ENUM_DEPRECATION_MSG)
    static constexpr thread_priority thread_priority_normal =
        thread_priority::normal;
    HPX_DEPRECATED_V(1, 6, HPX_THREAD_PRIORITY_UNSCOPED_ENUM_DEPRECATION_MSG)
    static constexpr thread_priority thread_priority_high_recursive =
        thread_priority::high_recursive;
    HPX_DEPRECATED_V(1, 6, HPX_THREAD_PRIORITY_UNSCOPED_ENUM_DEPRECATION_MSG)
    static constexpr thread_priority thread_priority_boost =
        thread_priority::boost;
    HPX_DEPRECATED_V(1, 6, HPX_THREAD_PRIORITY_UNSCOPED_ENUM_DEPRECATION_MSG)
    static constexpr thread_priority thread_priority_high =
        thread_priority::high;
    HPX_DEPRECATED_V(1, 6, HPX_THREAD_PRIORITY_UNSCOPED_ENUM_DEPRECATION_MSG)
    static constexpr thread_priority thread_priority_bound =
        thread_priority::bound;
    HPX_DEPRECATED_V(1, 6, HPX_THREAD_PRIORITY_UNSCOPED_ENUM_DEPRECATION_MSG)
    static constexpr thread_priority thread_priority_critical =
        thread_priority::critical;
#undef HPX_THREAD_PRIORITY_UNSCOPED_ENUM_DEPRECATION_MSG

    HPX_CORE_EXPORT std::ostream& operator<<(
        std::ostream& os, thread_priority const t);

    ////////////////////////////////////////////////////////////////////////////
    /// \brief Return the thread priority name.
    ///
    /// Get the readable string representing the name of the given thread_priority
    /// constant.
    ///
    /// \param this represents the thread priority.
    HPX_CORE_EXPORT char const* get_thread_priority_name(
        thread_priority priority);

    ///////////////////////////////////////////////////////////////////////////
    /// \enum thread_restart_state
    ///
    /// The \a thread_restart_state enumerator encodes the reason why a
    /// thread is being restarted
    enum class thread_restart_state
    {
        unknown = 0,
        signaled = 1,     ///< The thread has been signaled
        timeout = 2,      ///< The thread has been reactivated after a
                          ///< timeout
        terminate = 3,    ///< The thread needs to be terminated
        abort = 4         ///< The thread needs to be aborted
    };

#define HPX_THREAD_STATE_EX_UNSCOPED_ENUM_DEPRECATION_MSG                      \
    "The unscoped thread_state_ex_enum names are deprecated. Please use "      \
    "thread_restart_state::state instead."

    HPX_DEPRECATED_V(1, 6, HPX_THREAD_STATE_EX_UNSCOPED_ENUM_DEPRECATION_MSG)
    static constexpr thread_restart_state wait_unknown =
        thread_restart_state::unknown;
    HPX_DEPRECATED_V(1, 6, HPX_THREAD_STATE_EX_UNSCOPED_ENUM_DEPRECATION_MSG)
    static constexpr thread_restart_state wait_signaled =
        thread_restart_state::signaled;
    HPX_DEPRECATED_V(1, 6, HPX_THREAD_STATE_EX_UNSCOPED_ENUM_DEPRECATION_MSG)
    static constexpr thread_restart_state wait_timeout =
        thread_restart_state::timeout;
    HPX_DEPRECATED_V(1, 6, HPX_THREAD_STATE_EX_UNSCOPED_ENUM_DEPRECATION_MSG)
    static constexpr thread_restart_state wait_terminate =
        thread_restart_state::terminate;
    HPX_DEPRECATED_V(1, 6, HPX_THREAD_STATE_EX_UNSCOPED_ENUM_DEPRECATION_MSG)
    static constexpr thread_restart_state wait_abort =
        thread_restart_state::abort;
#undef HPX_THREAD_STATE_EX_UNSCOPED_ENUM_DEPRECATION_MSG

    HPX_CORE_EXPORT std::ostream& operator<<(
        std::ostream& os, thread_restart_state const t);

    /// Get the readable string representing the name of the given
    /// thread_restart_state constant.
    HPX_CORE_EXPORT char const* get_thread_state_ex_name(
        thread_restart_state state);

    /// \cond NOINTERNAL
    // special type storing both state in one tagged structure
    using thread_state =
        threads::detail::combined_tagged_state<thread_schedule_state,
            thread_restart_state>;
    /// \endcond

    /// Get the readable string representing the name of the given
    /// thread_state constant.
    HPX_CORE_EXPORT char const* get_thread_state_name(thread_state state);

    ///////////////////////////////////////////////////////////////////////////
    /// \enum thread_stacksize
    ///
    /// A \a thread_stacksize references any of the possible stack-sizes for
    /// HPX threads.
    enum class thread_stacksize
    {
        unknown = -1,
        small_ = 1,     ///< use small stack size (the underscore is to work
                        ///  around small being defined to char on Windows)
        medium = 2,     ///< use medium sized stack size
        large = 3,      ///< use large stack size
        huge = 4,       ///< use very large stack size
        nostack = 5,    ///< this thread does not suspend
                        ///< (does not need a stack)
        current = 6,    ///< use size of current thread's stack

        default_ = small_,    ///< use default stack size
        minimal = small_,     ///< use minimally stack size
        maximal = huge,       ///< use maximally stack size
    };

#define HPX_THREAD_STACKSIZE_UNSCOPED_ENUM_DEPRECATION_MSG                     \
    "The unscoped thread_stacksize names are deprecated. Please use "          \
    "thread_stacksize::size instead."

    HPX_DEPRECATED_V(1, 6, HPX_THREAD_STACKSIZE_UNSCOPED_ENUM_DEPRECATION_MSG)
    static constexpr thread_stacksize thread_stacksize_unknown =
        thread_stacksize::unknown;
    HPX_DEPRECATED_V(1, 6, HPX_THREAD_STACKSIZE_UNSCOPED_ENUM_DEPRECATION_MSG)
    static constexpr thread_stacksize thread_stacksize_small =
        thread_stacksize::small_;
    HPX_DEPRECATED_V(1, 6, HPX_THREAD_STACKSIZE_UNSCOPED_ENUM_DEPRECATION_MSG)
    static constexpr thread_stacksize thread_stacksize_medium =
        thread_stacksize::medium;
    HPX_DEPRECATED_V(1, 6, HPX_THREAD_STACKSIZE_UNSCOPED_ENUM_DEPRECATION_MSG)
    static constexpr thread_stacksize thread_stacksize_large =
        thread_stacksize::large;
    HPX_DEPRECATED_V(1, 6, HPX_THREAD_STACKSIZE_UNSCOPED_ENUM_DEPRECATION_MSG)
    static constexpr thread_stacksize thread_stacksize_huge =
        thread_stacksize::huge;
    HPX_DEPRECATED_V(1, 6, HPX_THREAD_STACKSIZE_UNSCOPED_ENUM_DEPRECATION_MSG)
    static constexpr thread_stacksize thread_stacksize_nostack =
        thread_stacksize::nostack;
    HPX_DEPRECATED_V(1, 6, HPX_THREAD_STACKSIZE_UNSCOPED_ENUM_DEPRECATION_MSG)
    static constexpr thread_stacksize thread_stacksize_current =
        thread_stacksize::current;
    HPX_DEPRECATED_V(1, 6, HPX_THREAD_STACKSIZE_UNSCOPED_ENUM_DEPRECATION_MSG)
    static constexpr thread_stacksize thread_stacksize_default =
        thread_stacksize::default_;
    HPX_DEPRECATED_V(1, 6, HPX_THREAD_STACKSIZE_UNSCOPED_ENUM_DEPRECATION_MSG)
    static constexpr thread_stacksize thread_stacksize_minimal =
        thread_stacksize::minimal;
    HPX_DEPRECATED_V(1, 6, HPX_THREAD_STACKSIZE_UNSCOPED_ENUM_DEPRECATION_MSG)
    static constexpr thread_stacksize thread_stacksize_maximal =
        thread_stacksize::maximal;
#undef HPX_THREAD_STACKSIZE_UNSCOPED_ENUM_DEPRECATION_MSG

    HPX_CORE_EXPORT std::ostream& operator<<(
        std::ostream& os, thread_stacksize const t);

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Returns the stack size name.
    ///
    /// Get the readable string representing the given stack size
    /// constant.
    ///
    /// \param size this represents the stack size
    HPX_CORE_EXPORT char const* get_stack_size_enum_name(thread_stacksize size);

    ///////////////////////////////////////////////////////////////////////////
    /// \enum thread_schedule_hint_mode
    ///
    /// The type of hint given when creating new tasks.
    enum class thread_schedule_hint_mode : std::int16_t
    {
        none = 0,
        thread = 1,
        numa = 2,
    };

#define HPX_THREAD_SCHEDULE_HINT_UNSCOPED_ENUM_DEPRECATION_MSG                 \
    "The unscoped thread_schedule_hint_mode names are deprecated. Please use " \
    "thread_schedule_hint_mode::hint instead."

    HPX_DEPRECATED_V(
        1, 6, HPX_THREAD_SCHEDULE_HINT_UNSCOPED_ENUM_DEPRECATION_MSG)
    static constexpr thread_schedule_hint_mode thread_schedule_hint_mode_none =
        thread_schedule_hint_mode::none;
    HPX_DEPRECATED_V(
        1, 6, HPX_THREAD_SCHEDULE_HINT_UNSCOPED_ENUM_DEPRECATION_MSG)
    static constexpr thread_schedule_hint_mode
        thread_schedule_hint_mode_thread = thread_schedule_hint_mode::thread;
    HPX_DEPRECATED_V(
        1, 6, HPX_THREAD_SCHEDULE_HINT_UNSCOPED_ENUM_DEPRECATION_MSG)
    static constexpr thread_schedule_hint_mode thread_schedule_hint_mode_numa =
        thread_schedule_hint_mode::numa;
#undef HPX_THREAD_SCHEDULE_HINT_UNSCOPED_ENUM_DEPRECATION_MSG

    ///////////////////////////////////////////////////////////////////////////
    struct thread_schedule_hint
    {
        constexpr thread_schedule_hint() noexcept
          : mode(thread_schedule_hint_mode::none)
          , hint(-1)
        {
        }

        constexpr explicit thread_schedule_hint(std::int16_t thread_hint)
          : mode(thread_schedule_hint_mode::thread)
          , hint(thread_hint)
        {
        }

        constexpr thread_schedule_hint(
            thread_schedule_hint_mode mode, std::int16_t hint) noexcept
          : mode(mode)
          , hint(hint)
        {
        }

        bool operator==(thread_schedule_hint const& rhs) const noexcept
        {
            return mode == rhs.mode && hint == rhs.hint;
        }

        bool operator!=(thread_schedule_hint const& rhs) const noexcept
        {
            return !(*this == rhs);
        }

        thread_schedule_hint_mode mode;
        std::int16_t hint;
    };
}}    // namespace hpx::threads
