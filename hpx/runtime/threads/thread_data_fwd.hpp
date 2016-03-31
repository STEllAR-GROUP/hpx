//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx/runtime/threads/thread_data_fwd.hpp

#if !defined(HPX_THREADS_THREAD_DATA_FWD_AUG_11_2015_0228PM)
#define HPX_THREADS_THREAD_DATA_FWD_AUG_11_2015_0228PM

#include <hpx/config.hpp>
#include <hpx/exception_fwd.hpp>
#include <hpx/runtime/threads/thread_enums.hpp>
#include <hpx/runtime/threads/coroutines/coroutine_fwd.hpp>
#include <hpx/util/unique_function.hpp>

#include <boost/intrusive_ptr.hpp>

namespace hpx { namespace threads
{
    /// \cond NOINTERNAL
    struct HPX_EXPORT threadmanager_base;
    class HPX_EXPORT thread_data;

    template <typename SchedulingPolicy>
    class HPX_EXPORT threadmanager_impl;

    typedef thread_state_enum thread_function_sig(thread_state_ex_enum);
    typedef util::unique_function_nonser<thread_function_sig>
        thread_function_type;

    class HPX_EXPORT executor;

    typedef coroutines::coroutine coroutine_type;

    typedef coroutines::detail::coroutine_self thread_self;
    typedef coroutines::detail::coroutine_impl thread_self_impl_type;

    typedef void * thread_id_repr_type;

    typedef boost::intrusive_ptr<thread_data> thread_id_type;

    HPX_API_EXPORT void intrusive_ptr_add_ref(thread_data* p);
    HPX_API_EXPORT void intrusive_ptr_release(thread_data* p);

    HPX_CONSTEXPR_OR_CONST thread_id_repr_type invalid_thread_id_repr = 0;
    thread_id_type const invalid_thread_id = thread_id_type();
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
    HPX_API_EXPORT thread_id_repr_type get_parent_id();

    /// The function \a get_parent_phase returns the HPX phase of the
    /// current thread's parent (or zero if the current thread is not a
    /// HPX thread).
    ///
    /// \note This function will return a meaningful value only if the
    ///       code was compiled with HPX_HAVE_THREAD_PARENT_REFERENCE
    ///       being defined.
    HPX_API_EXPORT std::size_t get_parent_phase();

    /// The function \a get_parent_locality_id returns the id of the locality of
    /// the current thread's parent (or zero if the current thread is not a
    /// HPX thread).
    ///
    /// \note This function will return a meaningful value only if the
    ///       code was compiled with HPX_HAVE_THREAD_PARENT_REFERENCE
    ///       being defined.
    HPX_API_EXPORT boost::uint32_t get_parent_locality_id();

    /// The function \a get_self_component_id returns the lva of the
    /// component the current thread is acting on
    ///
    /// \note This function will return a meaningful value only if the
    ///       code was compiled with HPX_HAVE_THREAD_TARGET_ADDRESS
    ///       being defined.
    HPX_API_EXPORT boost::uint64_t get_self_component_id();

    /// The function \a get_thread_manager returns a reference to the
    /// current thread manager.
    HPX_API_EXPORT threadmanager_base& get_thread_manager();

    /// The function \a get_thread_count returns the number of currently
    /// known threads.
    ///
    /// \note If state == unknown this function will not only return the
    ///       number of currently existing threads, but will add the number
    ///       of registered task descriptions (which have not been
    ///       converted into threads yet).
    HPX_API_EXPORT boost::int64_t get_thread_count(
        thread_state_enum state = unknown);

    /// \copydoc get_thread_count(thread_state_enum state)
    HPX_API_EXPORT boost::int64_t get_thread_count(
        thread_priority priority, thread_state_enum state = unknown);
}}

#endif

