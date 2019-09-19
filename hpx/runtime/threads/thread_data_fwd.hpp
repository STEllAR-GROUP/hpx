//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx/runtime/threads/thread_data_fwd.hpp

#if !defined(HPX_THREADS_THREAD_DATA_FWD_AUG_11_2015_0228PM)
#define HPX_THREADS_THREAD_DATA_FWD_AUG_11_2015_0228PM

#include <hpx/config.hpp>
#include <hpx/errors.hpp>
#include <hpx/coroutines/coroutine_fwd.hpp>
#include <hpx/coroutines/thread_enums.hpp>
#include <hpx/coroutines/thread_id_type.hpp>
#include <hpx/util_fwd.hpp>
#include <hpx/functional/function.hpp>
#include <hpx/functional/unique_function.hpp>
#if defined(HPX_HAVE_APEX)
// forward declare the APEX object
namespace apex { struct task_wrapper; }
typedef std::shared_ptr<apex::task_wrapper> apex_task_wrapper;
#endif

#include <cstddef>
#include <cstdint>
#include <utility>
#include <memory>

namespace hpx
{
    /// \cond NOINTERNAL
    class HPX_EXPORT thread;
    /// \endcond
}

namespace hpx { namespace threads
{
    /// \cond NOINTERNAL
    class HPX_EXPORT threadmanager;
    struct HPX_EXPORT topology;

    class HPX_EXPORT executor;

    typedef coroutines::coroutine coroutine_type;

    typedef coroutines::detail::coroutine_self thread_self;
    typedef coroutines::detail::coroutine_impl thread_self_impl_type;

    typedef std::pair<thread_state_enum, thread_id_type> thread_result_type;
    typedef thread_state_ex_enum thread_arg_type;

    typedef thread_result_type thread_function_sig(thread_arg_type);
    typedef util::unique_function_nonser<thread_function_sig> thread_function_type;
    /// \endcond

    ///////////////////////////////////////////////////////////////////////
    /// The function \a get_self returns a reference to the (OS thread
    /// specific) self reference to the current HPX thread.
    HPX_API_EXPORT thread_self& get_self();

    /// The function \a get_self_ptr returns a pointer to the (OS thread
    /// specific) self reference to the current HPX thread.
    HPX_API_EXPORT thread_self* get_self_ptr();

    /// The function \a get_ctx_ptr returns a pointer to the internal data
    /// associated with each coroutine.
    HPX_API_EXPORT thread_self_impl_type* get_ctx_ptr();

    /// The function \a get_self_ptr_checked returns a pointer to the (OS
    /// thread specific) self reference to the current HPX thread.
    HPX_API_EXPORT thread_self* get_self_ptr_checked(error_code& ec = throws);

    /// The function \a get_self_id returns the HPX thread id of the current
    /// thread (or zero if the current thread is not a HPX thread).
    HPX_API_EXPORT thread_id_type get_self_id();

    /// The function \a get_parent_id returns the HPX thread id of the
    /// current thread's parent (or zero if the current thread is not a
    /// HPX thread).
    ///
    /// \note This function will return a meaningful value only if the
    ///       code was compiled with HPX_HAVE_THREAD_PARENT_REFERENCE
    ///       being defined.
    HPX_API_EXPORT thread_id_type get_parent_id();

    /// The function \a get_parent_phase returns the HPX phase of the
    /// current thread's parent (or zero if the current thread is not a
    /// HPX thread).
    ///
    /// \note This function will return a meaningful value only if the
    ///       code was compiled with HPX_HAVE_THREAD_PARENT_REFERENCE
    ///       being defined.
    HPX_API_EXPORT std::size_t get_parent_phase();

    /// The function \a get_self_stacksize returns the stack size of the
    /// current thread (or zero if the current thread is not a HPX thread).
    HPX_API_EXPORT std::size_t get_self_stacksize();

    /// The function \a get_parent_locality_id returns the id of the locality of
    /// the current thread's parent (or zero if the current thread is not a
    /// HPX thread).
    ///
    /// \note This function will return a meaningful value only if the
    ///       code was compiled with HPX_HAVE_THREAD_PARENT_REFERENCE
    ///       being defined.
    HPX_API_EXPORT std::uint32_t get_parent_locality_id();

    /// The function \a get_self_component_id returns the lva of the
    /// component the current thread is acting on
    ///
    /// \note This function will return a meaningful value only if the
    ///       code was compiled with HPX_HAVE_THREAD_TARGET_ADDRESS
    ///       being defined.
    HPX_API_EXPORT std::uint64_t get_self_component_id();

    /// \cond NOINTERNAL
    // The function get_thread_manager returns a reference to the
    // current thread manager.
    HPX_API_EXPORT threadmanager& get_thread_manager();
    /// \endcond

    /// The function \a get_thread_count returns the number of currently
    /// known threads.
    ///
    /// \param state    [in] This specifies the thread-state for which the
    ///                 number of threads should be retrieved.
    ///
    /// \note If state == unknown this function will not only return the
    ///       number of currently existing threads, but will add the number
    ///       of registered task descriptions (which have not been
    ///       converted into threads yet).
    HPX_API_EXPORT std::int64_t get_thread_count(
        thread_state_enum state = unknown);

    /// The function \a get_thread_count returns the number of currently
    /// known threads.
    ///
    /// \param priority [in] This specifies the thread-priority for which the
    ///                 number of threads should be retrieved.
    /// \param state    [in] This specifies the thread-state for which the
    ///                 number of threads should be retrieved.
    ///
    /// \note If state == unknown this function will not only return the
    ///       number of currently existing threads, but will add the number
    ///       of registered task descriptions (which have not been
    ///       converted into threads yet).
    HPX_API_EXPORT std::int64_t get_thread_count(
        thread_priority priority, thread_state_enum state = unknown);

    /// The function \a enumerate_threads will invoke the given function \a f
    /// for each thread with a matching thread state.
    ///
    /// \param f        [in] The function which should be called for each
    ///                 matching thread. Returning 'false' from this function
    ///                 will stop the enumeration process.
    /// \param state    [in] This specifies the thread-state for which the
    ///                 threads should be enumerated.
    HPX_API_EXPORT bool enumerate_threads(
        util::function_nonser<bool(thread_id_type)> const& f,
        thread_state_enum state = unknown);

#if defined(HPX_HAVE_APEX)
    HPX_API_EXPORT apex_task_wrapper get_self_apex_data(void);
    HPX_API_EXPORT void set_self_apex_data(apex_task_wrapper data);
#endif
}}

#endif

