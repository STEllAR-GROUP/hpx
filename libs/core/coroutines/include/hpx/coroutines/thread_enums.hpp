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
#include <ostream>

namespace hpx { namespace threads {

    ///////////////////////////////////////////////////////////////////////////
    // clang-format off

    /// \enum thread_schedule_state
    ///
    /// The \a thread_schedule_state enumerator encodes the current state of a
    /// \a thread instance
    enum class thread_schedule_state : std::int8_t
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
    enum class thread_priority : std::int8_t
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
    enum class thread_restart_state : std::int8_t
    {
        unknown = 0,
        signaled = 1,     ///< The thread has been signaled
        timeout = 2,      ///< The thread has been reactivated after a
                          ///< timeout
        terminate = 3,    ///< The thread needs to be terminated
        abort = 4         ///< The thread needs to be aborted
    };

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
    enum class thread_stacksize : std::int8_t
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
    enum class thread_schedule_hint_mode : std::int8_t
    {
        /// A hint that leaves the choice of scheduling entirely up to the
        /// scheduler.
        none = 0,
        /// A hint that tells the scheduler to prefer scheduling a task on the
        /// local thread number associated with this hint. Local thread numbers
        /// are indexed from zero. It is up to the scheduler to decide how to
        /// interpret thread numbers that are larger than the number of threads
        /// available to the scheduler. Typically thread numbers will wrap
        /// around when too large.
        thread = 1,
        /// A hint that tells the scheduler to prefer scheduling a task on the
        /// NUMA domain associated with this hint. NUMA domains are indexed from
        /// zero. It is up to the scheduler to decide how to interpret NUMA
        /// domain indices that are larger than the number of available NUMA
        /// domains to the scheduler. Typically indices will wrap around when
        /// too large.
        numa = 2,
    };

    ///////////////////////////////////////////////////////////////////////////
    /// \brief A hint given to a scheduler to guide where a task should be
    /// scheduled.
    ///
    /// A scheduler is free to ignore the hint, or modify the hint to suit the
    /// resources available to the scheduler.
    struct thread_schedule_hint
    {
        /// Construct a default hint with mode thread_schedule_hint_mode::none.
        constexpr thread_schedule_hint() noexcept
          : hint(-1)
          , mode(thread_schedule_hint_mode::none)
        {
        }

        /// Construct a hint with mode thread_schedule_hint_mode::thread and the
        /// given hint as the local thread number.
        constexpr explicit thread_schedule_hint(
            std::int16_t thread_hint) noexcept
          : hint(thread_hint)
          , mode(thread_schedule_hint_mode::thread)
        {
        }

        /// Construct a hint with the given mode and hint. The numerical hint is
        /// unused when the mode is thread_schedule_hint_mode::none.
        constexpr thread_schedule_hint(
            thread_schedule_hint_mode mode, std::int16_t hint) noexcept
          : hint(hint)
          , mode(mode)
        {
        }

        /// \cond NOINTERNAL
        bool operator==(thread_schedule_hint const& rhs) const noexcept
        {
            return mode == rhs.mode && hint == rhs.hint;
        }

        bool operator!=(thread_schedule_hint const& rhs) const noexcept
        {
            return !(*this == rhs);
        }
        /// \endcond

        /// The hint associated with the mode. The interepretation of this hint
        /// depends on the given mode.
        std::int16_t hint;

        /// The mode of the scheduling hint.
        thread_schedule_hint_mode mode;
    };
}}    // namespace hpx::threads
