//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx/runtime/threads/thread_enums.hpp

#if !defined(HPX_THREAD_ENUMS_JUL_23_2015_0852PM)
#define HPX_THREAD_ENUMS_JUL_23_2015_0852PM

#include <hpx/config.hpp>
#include <hpx/runtime/threads/detail/tagged_thread_state.hpp>

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
        staged = 6          /*!< this is not a real thread state, but
                                 allows to reference staged task descriptions,
                                 which eventually will be converted into
                                 thread objects */
    };

    HPX_API_EXPORT char const* get_thread_state_name(thread_state_enum state);

    /// \ cond NODETAIL
    ///   Please note that if you change the value of threads::terminated
    ///   above, you will need to adjust do_call(dummy<1> = 1) in
    ///   util/coroutine/detail/coroutine_impl.hpp as well.
    /// \ endcond

    ///////////////////////////////////////////////////////////////////////
    /// \enum thread_priority
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

    typedef threads::detail::tagged_thread_state<thread_state_enum> thread_state;

    HPX_API_EXPORT char const* get_thread_priority_name(thread_priority priority);

    ///////////////////////////////////////////////////////////////////////////
    /// \enum thread_state_ex_enum
    ///
    /// The \a thread_state_ex_enum enumerator encodes the reason why a
    /// thread is being restarted
    enum thread_state_ex_enum
    {
        wait_unknown = -1,
        wait_signaled = 0,  ///< The thread has been signaled
        wait_timeout = 1,   ///< The thread has been reactivated after a timeout
        wait_terminate = 2, ///< The thread needs to be terminated
        wait_abort = 3      ///< The thread needs to be aborted
    };

    typedef threads::detail::tagged_thread_state<thread_state_ex_enum>
        thread_state_ex;

    ///////////////////////////////////////////////////////////////////////
    /// \enum thread_stacksize
    enum thread_stacksize
    {
        thread_stacksize_unknown = -1,
        thread_stacksize_small = 1,         ///< use small stack size
        thread_stacksize_medium = 2,        ///< use medium sized stack size
        thread_stacksize_large = 3,         ///< use large stack size
        thread_stacksize_huge = 4,          ///< use very large stack size
        thread_stacksize_nostack = 5,       ///< this thread does not suspend
                                            ///< (does not need a stack)

        thread_stacksize_default = thread_stacksize_small,  ///< use default stack size
        thread_stacksize_minimal = thread_stacksize_small,  ///< use minimally stack size
        thread_stacksize_maximal = thread_stacksize_huge,   ///< use maximally stack size
    };

    HPX_API_EXPORT char const* get_stack_size_name(std::ptrdiff_t size);
}}

#endif
