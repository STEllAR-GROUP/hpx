//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file threads_fwd.hpp

#ifndef HPX_RUNTIME_THREADS_FWD_HPP
#define HPX_RUNTIME_THREADS_FWD_HPP

#include <hpx/config.hpp>
#include <hpx/runtime/threads/thread_enums.hpp>
#include <hpx/util/coroutine/detail/default_context_impl.hpp>
#include <hpx/util/unique_function.hpp>

#include <boost/thread/mutex.hpp>

namespace hpx
{
    namespace util
    {
        namespace coroutines
        {
            namespace detail
            {
                template <typename Coroutine>
                class coroutine_self;

                template <typename CoroutineImpl>
                struct coroutine_allocator;
                template<typename CoroutineType, typename ContextImpl,
                    template <typename> class Heap>
                class coroutine_impl;
            }

            template<typename Signature,
                template <typename> class Heap,
                typename ContextImpl = detail::default_context_impl>
            class coroutine;
        }
    }

    /// \namespace threads
    ///
    /// The namespace \a thread-manager contains all the definitions required
    /// for the scheduling, execution and general management of \a
    /// hpx#threadmanager#thread's.
    namespace threads
    {
        namespace policies
        {
            struct scheduler_base;

            struct lockfree_fifo;
            struct lockfree_lifo;

            // multi priority scheduler with work-stealing
            template <typename Mutex = boost::mutex
                    , typename PendingQueuing = lockfree_fifo
                    , typename StagedQueuing = lockfree_fifo
                    , typename TerminatedQueuing = lockfree_lifo
                     >
            class HPX_EXPORT local_priority_queue_scheduler;

            // single priority scheduler with work-stealing
            template <typename Mutex = boost::mutex
                    , typename PendingQueuing = lockfree_fifo
                    , typename StagedQueuing = lockfree_fifo
                    , typename TerminatedQueuing = lockfree_lifo
                     >
            class HPX_EXPORT local_queue_scheduler;

#if defined(HPX_HAVE_PERIODIC_PRIORITY_SCHEDULER)
            template <typename Mutex = boost::mutex
                    , typename PendingQueuing = lockfree_fifo
                    , typename StagedQueuing = lockfree_fifo
                    , typename TerminatedQueuing = lockfree_lifo
                     >
            class HPX_EXPORT periodic_priority_queue_scheduler;
#endif

#if defined(HPX_HAVE_STATIC_PRIORITY_SCHEDULER)
            // multi priority scheduler with no work-stealing
            template <typename Mutex = boost::mutex
                    , typename PendingQueuing = lockfree_fifo
                    , typename StagedQueuing = lockfree_fifo
                    , typename TerminatedQueuing = lockfree_lifo
                     >
            class HPX_EXPORT static_priority_queue_scheduler;
#endif

#if defined(HPX_HAVE_STATIC_SCHEDULER)
            // single priority scheduler with no work-stealing
            template <typename Mutex = boost::mutex
                    , typename PendingQueuing = lockfree_fifo
                    , typename StagedQueuing = lockfree_fifo
                    , typename TerminatedQueuing = lockfree_lifo
                     >
            class HPX_EXPORT static_queue_scheduler;
#endif

#if defined(HPX_HAVE_THROTTLE_SCHEDULER)
            // single priority scheduler with work-stealing and throttling
            template <typename Mutex = boost::mutex
                    , typename PendingQueuing = lockfree_fifo
                    , typename StagedQueuing = lockfree_fifo
                    , typename TerminatedQueuing = lockfree_lifo
                     >
            class HPX_EXPORT throttle_queue_scheduler;
#endif

#if defined(HPX_HAVE_HIERARCHY_SCHEDULER)
            template <typename Mutex = boost::mutex
                    , typename PendingQueuing = lockfree_fifo
                    , typename StagedQueuing = lockfree_fifo
                    , typename TerminatedQueuing = lockfree_lifo
                     >
            class HPX_EXPORT hierarchy_scheduler;
#endif

            typedef local_priority_queue_scheduler<
                boost::mutex,
                lockfree_fifo, // FIFO pending queuing
                lockfree_fifo, // FIFO staged queuing
                lockfree_lifo  // LIFO terminated queuing
            > fifo_priority_queue_scheduler;

#if defined(HPX_HAVE_ABP_SCHEDULER)
            struct lockfree_abp_fifo;
            struct lockfree_abp_lifo;

            typedef local_priority_queue_scheduler<
                boost::mutex,
                lockfree_abp_fifo, // FIFO + ABP pending queuing
                lockfree_abp_fifo, // FIFO + ABP staged queuing
                lockfree_lifo  // LIFO terminated queuing
            > abp_fifo_priority_queue_scheduler;
#endif

            // define the default scheduler to use
            typedef fifo_priority_queue_scheduler queue_scheduler;

            class HPX_EXPORT callback_notifier;
        }

        struct HPX_EXPORT threadmanager_base;
        class HPX_EXPORT thread_data_base;
        class HPX_EXPORT thread_data;

        template <typename SchedulingPolicy>
        class HPX_EXPORT threadmanager_impl;

        typedef thread_state_enum thread_function_sig(thread_state_ex_enum);
        typedef util::unique_function_nonser<thread_function_sig>
            thread_function_type;

        class HPX_EXPORT executor;

        ///////////////////////////////////////////////////////////////////////
        /// \ cond NODETAIL
        namespace detail
        {
            template <typename CoroutineImpl> struct coroutine_allocator;
        }
        /// \ endcond
        typedef util::coroutines::coroutine<
            thread_function_sig, detail::coroutine_allocator> coroutine_type;

        typedef util::coroutines::detail::coroutine_self<coroutine_type>
            thread_self;
        typedef
            util::coroutines::detail::coroutine_impl<
                coroutine_type
              , util::coroutines::detail::default_context_impl
              , detail::coroutine_allocator
            >
            thread_self_impl_type;
        typedef void * thread_id_repr_type;

        typedef boost::intrusive_ptr<thread_data_base> thread_id_type;

        HPX_EXPORT void intrusive_ptr_add_ref(thread_data_base* p);
        HPX_EXPORT void intrusive_ptr_release(thread_data_base* p);

        ///////////////////////////////////////////////////////////////////////
        /// \ cond NODETAIL
        BOOST_CONSTEXPR_OR_CONST thread_id_repr_type invalid_thread_id_repr = 0;
        thread_id_type const invalid_thread_id = thread_id_type();
        /// \ endcond

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
    }
}

#endif
