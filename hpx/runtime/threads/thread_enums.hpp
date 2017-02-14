//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx/runtime/threads/thread_enums.hpp

#if !defined(HPX_THREAD_ENUMS_JUL_23_2015_0852PM)
#define HPX_THREAD_ENUMS_JUL_23_2015_0852PM

#include <hpx/config.hpp>
#include <hpx/runtime/threads/detail/combined_tagged_state.hpp>

#include <cstddef>

namespace hpx { namespace threads
{
    ///////////////////////////////////////////////////////////////////////////
    /// \enum thread_state_enum
    ///
    /// The \a thread_state_enum enumerator encodes the current state of a
    /// \a thread instance
    enum thread_state_enum
    {
        unknown = 0,
        active = 1,         /*!< thread is currently active (running,
                                 has resources) */
        pending = 2,        /*!< thread is pending (ready to run, but
                                 no hardware resource available) */
        suspended = 3,      /*!< thread has been suspended (waiting for
                                 synchronization event, but still
                                 known and under control of the
                                 thread-manager) */
        depleted = 4,       /*!< thread has been depleted (deeply
                                 suspended, it is not known to the
                                 thread-manager) */
        terminated = 5,     /*!< thread has been stopped an may be
                                 garbage collected */
        staged = 6,         /*!< this is not a real thread state, but
                                 allows to reference staged task descriptions,
                                 which eventually will be converted into
                                 thread objects */
        pending_do_not_schedule = 7, /*< this is not a real thread state,
                                 but allows to create a thread in pending state
                                 without scheduling it (internal, do not use) */
        pending_boost = 8   /*< this is not a real thread state,
                                 but allows to suspend a thread in pending state
                                 without high priority rescheduling */
    };

    /// Get the readable string representing the name of the given
    /// thread_state constant.
    HPX_API_EXPORT char const* get_thread_state_name(thread_state_enum state);

    ///////////////////////////////////////////////////////////////////////////
    /// This enumeration lists all possible thread-priorities for HPX threads.
    ///
    enum thread_priority
    {
        thread_priority_unknown = -1,
        thread_priority_default = 0,      ///< use default priority
        thread_priority_low = 1,          ///< low thread priority
        thread_priority_normal = 2,       ///< normal thread priority (default)
        thread_priority_critical = 3,     ///< high thread priority
        thread_priority_boost = 4         ///< high thread priority for first
                                          ///< invocation, normal afterwards
    };

    /// Get the readable string representing the name of the given thread_priority
    /// constant.
    HPX_API_EXPORT char const* get_thread_priority_name(thread_priority priority);

    ///////////////////////////////////////////////////////////////////////////
    /// \enum thread_state_ex_enum
    ///
    /// The \a thread_state_ex_enum enumerator encodes the reason why a
    /// thread is being restarted
    enum thread_state_ex_enum
    {
        wait_unknown = 0,
        wait_signaled = 1,  ///< The thread has been signaled
        wait_timeout = 2,   ///< The thread has been reactivated after a timeout
        wait_terminate = 3, ///< The thread needs to be terminated
        wait_abort = 4      ///< The thread needs to be aborted
    };

    /// Get the readable string representing the name of the given
    /// thread_state_ex_enum constant.
    HPX_API_EXPORT char const* get_thread_state_ex_name(thread_state_ex_enum state);

    /// \cond NOINTERNAL
    // special type storing both state in one tagged structure
    typedef threads::detail::combined_tagged_state<
            thread_state_enum, thread_state_ex_enum
        > thread_state;
    /// \endcond

    /// Get the readable string representing the name of the given
    /// thread_state constant.
    HPX_API_EXPORT char const* get_thread_state_name(thread_state state);

    ///////////////////////////////////////////////////////////////////////////
    /// \enum thread_stacksize
    ///
    /// A \a thread_stacksize references any of the possible stack-sizes for
    /// HPX threads.
    enum thread_stacksize
    {
        thread_stacksize_unknown = -1,
        thread_stacksize_small = 1,         ///< use small stack size
        thread_stacksize_medium = 2,        ///< use medium sized stack size
        thread_stacksize_large = 3,         ///< use large stack size
        thread_stacksize_huge = 4,          ///< use very large stack size

        thread_stacksize_current = 5,      ///< use size of current thread's stack

        thread_stacksize_default = thread_stacksize_small,  ///< use default stack size
        thread_stacksize_minimal = thread_stacksize_small,  ///< use minimally stack size
        thread_stacksize_maximal = thread_stacksize_huge,   ///< use maximally stack size
    };

    /// Get the readable string representing the given stack size
    /// constant.
    HPX_API_EXPORT char const* get_stack_size_name(std::ptrdiff_t size);
}}

#endif
