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

namespace hpx { namespace threads {

    ///////////////////////////////////////////////////////////////////////////
    // clang-format off

    /// \enum thread_state_enum
    ///
    /// The \a thread_state_enum enumerator encodes the current state of a
    /// \a thread instance
    enum thread_state_enum
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

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Returns the name of the given state
    ///
    /// Get the readable string representing the name of the given
    /// thread_state constant.
    ///
    /// \param state this represents the thread state.
    HPX_CORE_EXPORT char const* get_thread_state_name(thread_state_enum state);

    ///////////////////////////////////////////////////////////////////////////
    // clang-format off

    /// This enumeration lists all possible thread-priorities for HPX threads.
    ///
    enum thread_priority
    {
        thread_priority_unknown = -1,
        thread_priority_default = 0, /*!< Will assign the priority of the
            task to the default (normal) priority. */
        thread_priority_low = 1,     /*!< Task goes onto a special low
            priority queue and will not be executed until all high/normal
            priority tasks are done, even if they are added after the low
            priority task. */
        thread_priority_normal = 2,  /*!< Task will be executed when it is
            taken from the normal priority queue, this is usually a first
            in-first-out ordering of tasks (depending on scheduler choice).
            This is the default priority. */
        thread_priority_high_recursive = 3, /*!< The task is a high priority
            task and any child tasks spawned by this task will be made high
            priority as well - unless they are specifically flagged as non
            default priority. */
        thread_priority_boost = 4, /*!< Same as \a thread_priority_high
            except that the thread will fall back to \a thread_priority_normal
            if resumed after being suspended. */
        thread_priority_high = 5,  /*!< Task goes onto a special high
            priority queue and will be executed before normal/low priority
            tasks are taken (some schedulers modify the behavior slightly and
            the documentation for those should be consulted). */
        thread_priority_bound = 6,          /*!< Task goes onto a special
            high priority queue and will never be stolen by another thread
            after initial assignment. This should be used for thread placement
            tasks such as OpenMP type for loops. */

        /// \cond NOINTERNAL
        // obsolete, kept for compatibility only
        thread_priority_critical = thread_priority_high_recursive,
        /// \endcond
    };
    // clang-format on

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
    /// \enum thread_state_ex_enum
    ///
    /// The \a thread_state_ex_enum enumerator encodes the reason why a
    /// thread is being restarted
    enum thread_state_ex_enum
    {
        wait_unknown = 0,
        wait_signaled = 1,     ///< The thread has been signaled
        wait_timeout = 2,      ///< The thread has been reactivated after a
                               ///< timeout
        wait_terminate = 3,    ///< The thread needs to be terminated
        wait_abort = 4         ///< The thread needs to be aborted
    };

    /// Get the readable string representing the name of the given
    /// thread_state_ex_enum constant.
    HPX_CORE_EXPORT char const* get_thread_state_ex_name(
        thread_state_ex_enum state);

    /// \cond NOINTERNAL
    // special type storing both state in one tagged structure
    using thread_state =
        threads::detail::combined_tagged_state<thread_state_enum,
            thread_state_ex_enum>;
    /// \endcond

    /// Get the readable string representing the name of the given
    /// thread_state constant.
    HPX_CORE_EXPORT char const* get_thread_state_name(thread_state state);

    ///////////////////////////////////////////////////////////////////////////
    /// \enum thread_stacksize
    ///
    /// A \a thread_stacksize references any of the possible stack-sizes for
    /// HPX threads.
    enum thread_stacksize
    {
        thread_stacksize_unknown = -1,
        thread_stacksize_small = 1,      ///< use small stack size
        thread_stacksize_medium = 2,     ///< use medium sized stack size
        thread_stacksize_large = 3,      ///< use large stack size
        thread_stacksize_huge = 4,       ///< use very large stack size
        thread_stacksize_nostack = 5,    ///< this thread does not suspend
                                         ///< (does not need a stack)
        thread_stacksize_current = 6,    ///< use size of current thread's stack

        thread_stacksize_default =
            thread_stacksize_small,    ///< use default stack size
        thread_stacksize_minimal =
            thread_stacksize_small,    ///< use minimally stack size
        thread_stacksize_maximal =
            thread_stacksize_huge,    ///< use maximally stack size
    };

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
    enum thread_schedule_hint_mode : std::int16_t
    {
        thread_schedule_hint_mode_none = 0,
        thread_schedule_hint_mode_thread = 1,
        thread_schedule_hint_mode_numa = 2,
    };

    ///////////////////////////////////////////////////////////////////////////
    struct thread_schedule_hint
    {
        constexpr thread_schedule_hint() noexcept
          : mode(thread_schedule_hint_mode_none)
          , hint(-1)
        {
        }

        constexpr explicit thread_schedule_hint(std::int16_t thread_hint)
          : mode(thread_schedule_hint_mode_thread)
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
