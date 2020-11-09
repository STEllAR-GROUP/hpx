//  Copyright (c) 2007-2015 Hartmut Kaiser
//  Copyright (c)      2018 Thomas Heller
//  Copyright (c)      2011 Bryce Lelbach
//  Copyright (c) 2008-2009 Chirag Dekate, Anshul Tandon
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/config/asio.hpp>
#include <hpx/assert.hpp>
#include <hpx/functional/function.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/threading_base/scheduler_base.hpp>
#include <hpx/threading_base/thread_data.hpp>
#include <hpx/threading_base/thread_pool_base.hpp>

#include <boost/asio/io_service.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace hpx { namespace threads {
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Create a new \a thread using the given function as the work to
    ///        be executed.
    ///
    /// \param func       [in] The function to be executed as the thread-function.
    ///                   This function has to expose the minimal low level
    ///                   HPX-thread interface, i.e. it takes one argument (a
    ///                   \a threads#thread_restart_state). The thread will be
    ///                   terminated after the function returns.
    ///
    /// \note All other arguments are equivalent to those of the function
    ///       \a threads#register_thread_plain
    ///
    namespace detail {
        template <typename F>
        struct thread_function
        {
            F f;

            inline threads::thread_result_type operator()(
                threads::thread_arg_type)
            {
                // execute the actual thread function
                f(threads::thread_restart_state::signaled);

                // Verify that there are no more registered locks for this
                // OS-thread. This will throw if there are still any locks
                // held.
                util::force_error_on_lock();

                // run and free all registered exit functions for this thread
                auto* p = get_self_id_data();
                p->run_thread_exit_callbacks();
                p->free_thread_exit_callbacks();

                return threads::thread_result_type(
                    threads::thread_schedule_state::terminated,
                    threads::invalid_thread_id);
            }
        };

        template <typename F>
        struct thread_function_nullary
        {
            F f;

            inline threads::thread_result_type operator()(
                threads::thread_arg_type)
            {
                // execute the actual thread function
                f();

                // Verify that there are no more registered locks for this
                // OS-thread. This will throw if there are still any locks
                // held.
                util::force_error_on_lock();

                // run and free all registered exit functions for this thread
                auto* p = get_self_id_data();
                p->run_thread_exit_callbacks();
                p->free_thread_exit_callbacks();

                return threads::thread_result_type(
                    threads::thread_schedule_state::terminated,
                    threads::invalid_thread_id);
            }
        };
    }    // namespace detail

    template <typename F>
    thread_function_type make_thread_function(F&& f)
    {
        return {detail::thread_function<typename std::decay<F>::type>{
            std::forward<F>(f)}};
    }

    template <typename F>
    thread_function_type make_thread_function_nullary(F&& f)
    {
        return {detail::thread_function_nullary<typename std::decay<F>::type>{
            std::forward<F>(f)}};
    }

    namespace detail {
        using get_default_pool_type =
            util::function_nonser<thread_pool_base*()>;
        HPX_CORE_EXPORT void set_get_default_pool(get_default_pool_type f);
        HPX_CORE_EXPORT thread_pool_base* get_self_or_default_pool();

        using get_default_timer_service_type =
            util::function_nonser<boost::asio::io_service*()>;
        HPX_CORE_EXPORT void set_get_default_timer_service(
            get_default_timer_service_type f);
        HPX_CORE_EXPORT boost::asio::io_service* get_default_timer_service();
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Create a new \a thread using the given data.
    ///
    /// \param data       [in] The data to use for creating the thread.
    /// \param pool       [in] The thread pool to use for launching the work.
    /// \param ec         [in,out] This represents the error status on exit,
    ///                   if this is pre-initialized to \a hpx#throws
    ///                   the function will throw on error instead.
    ///
    /// \returns This function will return the internal id of the newly created
    ///          HPX-thread.
    ///
    /// \throws invalid_status if the runtime system has not been started yet.
    ///
    /// \note             As long as \a ec is not pre-initialized to
    ///                   \a hpx#throws this function doesn't
    ///                   throw but returns the result code using the
    ///                   parameter \a ec. Otherwise it throws an instance
    ///                   of hpx#exception.
    inline threads::thread_id_type register_thread(
        threads::thread_init_data& data, threads::thread_pool_base* pool,
        error_code& ec = throws)
    {
        HPX_ASSERT(pool);
        data.run_now = true;
        threads::thread_id_type id = threads::invalid_thread_id;
        pool->create_thread(data, id, ec);
        return id;
    }

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Create a new \a thread using the given data on the same thread
    ///        pool as the calling thread, or on the default thread pool if
    ///        not on an HPX thread.
    ///
    /// \param data       [in] The data to use for creating the thread.
    /// \param ec         [in,out] This represents the error status on exit,
    ///                   if this is pre-initialized to \a hpx#throws
    ///                   the function will throw on error instead.
    ///
    /// \returns This function will return the internal id of the newly created
    ///          HPX-thread.
    ///
    /// \throws invalid_status if the runtime system has not been started yet.
    ///
    /// \note             As long as \a ec is not pre-initialized to
    ///                   \a hpx#throws this function doesn't
    ///                   throw but returns the result code using the
    ///                   parameter \a ec. Otherwise it throws an instance
    ///                   of hpx#exception.
    inline threads::thread_id_type register_thread(
        threads::thread_init_data& data, error_code& ec = throws)
    {
        return register_thread(data, detail::get_self_or_default_pool(), ec);
    }

    /// \brief Create a new work item using the given data.
    ///
    /// \param data       [in] The data to use for creating the thread.
    /// \param pool       [in] The thread pool to use for launching the work.
    /// \param ec         [in,out] This represents the error status on exit,
    ///                   if this is pre-initialized to \a hpx#throws
    ///                   the function will throw on error instead.
    ///
    /// \throws invalid_status if the runtime system has not been started yet.
    ///
    /// \note             As long as \a ec is not pre-initialized to
    ///                   \a hpx#throws this function doesn't
    ///                   throw but returns the result code using the
    ///                   parameter \a ec. Otherwise it throws an instance
    ///                   of hpx#exception.
    inline void register_work(threads::thread_init_data& data,
        threads::thread_pool_base* pool, error_code& ec = throws)
    {
        HPX_ASSERT(pool);
        data.run_now = false;
        pool->create_work(data, ec);
    }

    /// \brief Create a new work item using the given data on the same thread
    ///        pool as the calling thread, or on the default thread pool if
    ///        not on an HPX thread.
    ///
    /// \param data       [in] The data to use for creating the thread.
    /// \param ec         [in,out] This represents the error status on exit,
    ///                   if this is pre-initialized to \a hpx#throws
    ///                   the function will throw on error instead.
    ///
    /// \throws invalid_status if the runtime system has not been started yet.
    ///
    /// \note             As long as \a ec is not pre-initialized to
    ///                   \a hpx#throws this function doesn't
    ///                   throw but returns the result code using the
    ///                   parameter \a ec. Otherwise it throws an instance
    ///                   of hpx#exception.
    inline void register_work(
        threads::thread_init_data& data, error_code& ec = throws)
    {
        register_work(data, detail::get_self_or_default_pool(), ec);
    }

#if defined(HPX_HAVE_REGISTER_THREAD_OVERLOADS_COMPATIBILITY)
    inline threads::thread_id_type register_thread_plain(
        threads::thread_pool_base* pool, threads::thread_init_data& data,
        threads::thread_schedule_state initial_state =
            threads::thread_schedule_state::pending,
        bool run_now = true, error_code& ec = throws)
    {
        HPX_ASSERT(pool);
        threads::thread_id_type id = threads::invalid_thread_id;
        data.initial_state = initial_state;
        data.run_now = run_now;
        pool->create_thread(data, id, ec);
        return id;
    }

    inline threads::thread_id_type register_non_suspendable_thread_plain(
        threads::thread_pool_base* pool, threads::thread_init_data& data,
        threads::thread_schedule_state initial_state =
            threads::thread_schedule_state::pending,
        bool run_now = true, error_code& ec = throws)
    {
        HPX_ASSERT(pool);
        data.stacksize = threads::thread_stacksize::nostack;
        return register_thread_plain(pool, data, initial_state, run_now, ec);
    }

    inline threads::thread_id_type register_thread_plain(
        threads::thread_pool_base* pool, threads::thread_function_type&& func,
        util::thread_description const& description =
            util::thread_description(),
        threads::thread_schedule_state initial_state =
            threads::thread_schedule_state::pending,
        bool run_now = true,
        threads::thread_priority priority = threads::thread_priority::normal,
        threads::thread_schedule_hint schedulehint =
            threads::thread_schedule_hint(),
        threads::thread_stacksize stacksize =
            threads::thread_stacksize::default_,
        error_code& ec = throws)
    {
        util::thread_description d = description ?
            description :
            util::thread_description(func, "register_thread_plain");

        HPX_ASSERT(pool);
        threads::thread_init_data data(
            std::move(func), d, priority, schedulehint, stacksize);

        return register_thread_plain(pool, data, initial_state, run_now, ec);
    }

    inline threads::thread_id_type register_non_suspendable_thread_plain(
        threads::thread_pool_base* pool, threads::thread_function_type&& func,
        util::thread_description const& description =
            util::thread_description(),
        threads::thread_schedule_state initial_state =
            threads::thread_schedule_state::pending,
        bool run_now = true,
        threads::thread_priority priority = threads::thread_priority::normal,
        threads::thread_schedule_hint schedulehint =
            threads::thread_schedule_hint(),
        error_code& ec = throws)
    {
        return register_thread_plain(pool, std::move(func), description,
            initial_state, run_now, priority, schedulehint,
            threads::thread_stacksize::nostack, ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Create a new \a thread using the given data.
    ///
    /// \note This function is completely equivalent to the first overload
    ///       of threads#register_thread_plain above, except that part of the
    ///       parameters are passed as members of the threads#thread_init_data
    ///       object.
    ///
    inline threads::thread_id_type register_thread_plain(
        threads::thread_init_data& data,
        threads::thread_schedule_state initial_state =
            threads::thread_schedule_state::pending,
        bool run_now = true, error_code& ec = throws)
    {
        return register_thread_plain(detail::get_self_or_default_pool(), data,
            initial_state, run_now, ec);
    }

    inline threads::thread_id_type register_non_suspendable_thread_plain(
        threads::thread_init_data& data,
        threads::thread_schedule_state initial_state =
            threads::thread_schedule_state::pending,
        bool run_now = true, error_code& ec = throws)
    {
        return register_non_suspendable_thread_plain(
            detail::get_self_or_default_pool(), data, initial_state, run_now,
            ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Create a new \a thread using the given function as the work to
    ///        be executed.
    ///
    /// \param func       [in] The function to be executed as the thread-function.
    ///                   This function has to expose the minimal low level
    ///                   HPX-thread interface, i.e. it takes one argument (a
    ///                   \a threads#thread_restart_state) and returns a
    ///                   \a threads#thread_schedule_state.
    /// \param description [in] A optional string describing the newly created
    ///                   thread. This is useful for debugging and logging
    ///                   purposes as this string will be inserted in the logs.
    /// \param initial_state [in] The thread state the newly created thread
    ///                   should have. If this is not given it defaults to
    ///                   \a threads#pending, which means that the new thread
    ///                   will be scheduled to run as soon as it is created.
    /// \param run_now    [in] If this is set to `true` the thread object will
    ///                   be actually immediately created. Otherwise the
    ///                   thread-manager creates a work-item description, which
    ///                   will result in creating a thread object later (if
    ///                   no work is available any more). The default is to
    ///                   immediately create the thread object.
    /// \param priority   [in] This is the priority the newly created HPX-thread
    ///                   should be executed with. The default is \a
    ///                   threads#thread_priority::normal. This parameter is not
    ///                   guaranteed to be taken into account as it depends on
    ///                   the used scheduling policy whether priorities are
    ///                   supported in the first place.
    /// \param os_thread  [in] The number of the shepherd thread the newly
    ///                   created HPX-thread should run on. If this is given it
    ///                   will be no more than a hint in any case, mainly
    ///                   because even if the HPX-thread gets scheduled on the
    ///                   queue of the requested shepherd thread, it still can
    ///                   be stolen by another shepherd thread. If this is not
    ///                   given, the system will select a shepherd thread.
    /// \param ec         [in,out] This represents the error status on exit,
    ///                   if this is pre-initialized to \a hpx#throws
    ///                   the function will throw on error instead.
    ///
    /// \returns This function will return the internal id of the newly created
    ///          HPX-thread or threads#invalid_thread_id (if run_now is set to
    ///          `false`).
    ///
    /// \note The value returned by the thread function will be interpreted by
    ///       the thread manager as the new thread state the executed HPX-thread
    ///       needs to be switched to. Normally, HPX-threads will either return
    ///       \a threads#terminated (if the thread should be destroyed) or
    ///       \a threads#suspended (if the thread needs to be suspended because
    ///       it is waiting for an external event to happen). The external
    ///       event will set the state of the thread back to pending, which
    ///       will re-schedule the HPX-thread.
    ///
    /// \throws invalid_status if the runtime system has not been started yet.
    ///
    ///
    /// \note             As long as \a ec is not pre-initialized to
    ///                   \a hpx#throws this function doesn't
    ///                   throw but returns the result code using the
    ///                   parameter \a ec. Otherwise it throws an instance
    ///                   of hpx#exception.
    inline threads::thread_id_type register_thread_plain(
        threads::thread_function_type&& func,
        util::thread_description const& description =
            util::thread_description(),
        threads::thread_schedule_state initial_state =
            threads::thread_schedule_state::pending,
        bool run_now = true,
        threads::thread_priority priority = threads::thread_priority::normal,
        threads::thread_schedule_hint schedulehint =
            threads::thread_schedule_hint(),
        threads::thread_stacksize stacksize =
            threads::thread_stacksize::default_,
        error_code& ec = throws)
    {
        return register_thread_plain(detail::get_self_or_default_pool(),
            std::move(func), description, initial_state, run_now, priority,
            schedulehint, stacksize, ec);
    }

    inline threads::thread_id_type register_non_suspendable_thread_plain(
        threads::thread_function_type&& func,
        util::thread_description const& description =
            util::thread_description(),
        threads::thread_schedule_state initial_state =
            threads::thread_schedule_state::pending,
        bool run_now = true,
        threads::thread_priority priority = threads::thread_priority::normal,
        threads::thread_schedule_hint schedulehint =
            threads::thread_schedule_hint(),
        error_code& ec = throws)
    {
        return register_non_suspendable_thread_plain(
            detail::get_self_or_default_pool(), std::move(func), description,
            initial_state, run_now, priority, schedulehint, ec);
    }

    template <typename F>
    threads::thread_id_type register_thread(F&& func,
        util::thread_description const& description =
            util::thread_description(),
        threads::thread_schedule_state initial_state =
            threads::thread_schedule_state::pending,
        bool run_now = true,
        threads::thread_priority priority = threads::thread_priority::normal,
        threads::thread_schedule_hint os_thread =
            threads::thread_schedule_hint(),
        threads::thread_stacksize stacksize =
            threads::thread_stacksize::default_,
        error_code& ec = throws)
    {
        threads::thread_function_type thread_func(
            detail::thread_function<typename std::decay<F>::type>{
                std::forward<F>(func)});
        return register_thread_plain(std::move(thread_func), description,
            initial_state, run_now, priority, os_thread, stacksize, ec);
    }

    template <typename F>
    threads::thread_id_type register_non_suspendable_thread(F&& func,
        util::thread_description const& description =
            util::thread_description(),
        threads::thread_schedule_state initial_state =
            threads::thread_schedule_state::pending,
        bool run_now = true,
        threads::thread_priority priority = threads::thread_priority::normal,
        threads::thread_schedule_hint os_thread =
            threads::thread_schedule_hint(),
        error_code& ec = throws)
    {
        threads::thread_function_type thread_func(
            detail::thread_function<typename std::decay<F>::type>{
                std::forward<F>(func)});
        return register_non_suspendable_thread_plain(std::move(thread_func),
            description, initial_state, run_now, priority, os_thread, ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Create a new \a thread using the given function as the work to
    ///        be executed.
    ///
    /// \param func       [in] The function to be executed as the thread-function.
    ///                   This function has to expose the minimal low level
    ///                   HPX-thread interface, i.e. it takes no arguments. The
    ///                   thread will be terminated after the function returns.
    ///
    /// \note All other arguments are equivalent to those of the function
    ///       \a threads#register_thread_plain
    ///
    template <typename F>
    threads::thread_id_type register_thread_nullary(F&& func,
        util::thread_description const& description =
            util::thread_description(),
        threads::thread_schedule_state initial_state =
            threads::thread_schedule_state::pending,
        bool run_now = true,
        threads::thread_priority priority = threads::thread_priority::normal,
        threads::thread_schedule_hint os_thread =
            threads::thread_schedule_hint(),
        threads::thread_stacksize stacksize =
            threads::thread_stacksize::default_,
        error_code& ec = throws)
    {
        threads::thread_function_type thread_func(
            detail::thread_function_nullary<typename std::decay<F>::type>{
                std::forward<F>(func)});
        return register_thread_plain(std::move(thread_func), description,
            initial_state, run_now, priority, os_thread, stacksize, ec);
    }

    template <typename F>
    threads::thread_id_type register_non_suspendable_thread_nullary(F&& func,
        util::thread_description const& description =
            util::thread_description(),
        threads::thread_schedule_state initial_state =
            threads::thread_schedule_state::pending,
        bool run_now = true,
        threads::thread_priority priority = threads::thread_priority::normal,
        threads::thread_schedule_hint os_thread =
            threads::thread_schedule_hint(),
        error_code& ec = throws)
    {
        return register_thread_nullary(std::forward<F>(func), description,
            initial_state, run_now, priority, os_thread,
            threads::thread_stacksize::nostack, ec);
    }

    template <typename F>
    threads::thread_id_type register_thread_nullary(
        threads::thread_pool_base* pool, F&& func,
        util::thread_description const& description =
            util::thread_description(),
        threads::thread_schedule_state initial_state =
            threads::thread_schedule_state::pending,
        bool run_now = true,
        threads::thread_priority priority = threads::thread_priority::normal,
        threads::thread_schedule_hint os_thread =
            threads::thread_schedule_hint(),
        threads::thread_stacksize stacksize =
            threads::thread_stacksize::default_,
        error_code& ec = throws)
    {
        threads::thread_function_type thread_func(
            detail::thread_function_nullary<typename std::decay<F>::type>{
                std::forward<F>(func)});
        return register_thread_plain(pool, std::move(thread_func), description,
            initial_state, run_now, priority, os_thread, stacksize, ec);
    }

    template <typename F>
    threads::thread_id_type register_non_suspendable_thread_nullary(
        threads::thread_pool_base* pool, F&& func,
        util::thread_description const& description =
            util::thread_description(),
        threads::thread_schedule_state initial_state =
            threads::thread_schedule_state::pending,
        bool run_now = true,
        threads::thread_priority priority = threads::thread_priority::normal,
        threads::thread_schedule_hint os_thread =
            threads::thread_schedule_hint(),
        error_code& ec = throws)
    {
        threads::thread_function_type thread_func(
            detail::thread_function_nullary<typename std::decay<F>::type>{
                std::forward<F>(func)});
        return register_non_suspendable_thread_plain(pool,
            std::move(thread_func), description, initial_state, run_now,
            priority, os_thread, ec);
    }

    inline void register_work_plain(threads::thread_pool_base* pool,
        threads::thread_init_data& data,
        threads::thread_schedule_state initial_state =
            threads::thread_schedule_state::pending,
        error_code& ec = throws)
    {
        HPX_ASSERT(pool);
        data.initial_state = initial_state;
        pool->create_work(data, ec);
    }

    inline void register_non_suspendable_work_plain(
        threads::thread_pool_base* pool, threads::thread_init_data& data,
        threads::thread_schedule_state initial_state =
            threads::thread_schedule_state::pending,
        error_code& ec = throws)
    {
        HPX_ASSERT(pool);
        data.stacksize = threads::thread_stacksize::nostack;
        register_work_plain(pool, data, initial_state, ec);
    }

    inline void register_work_plain(threads::thread_pool_base* pool,
        threads::thread_function_type&& func,
        util::thread_description const& description =
            util::thread_description(),
        threads::thread_schedule_state initial_state =
            threads::thread_schedule_state::pending,
        threads::thread_priority priority = threads::thread_priority::normal,
        threads::thread_schedule_hint schedulehint =
            threads::thread_schedule_hint(),
        threads::thread_stacksize stacksize =
            threads::thread_stacksize::default_,
        error_code& ec = throws)
    {
        util::thread_description d = description ?
            description :
            util::thread_description(func, "register_work_plain");

        HPX_ASSERT(pool);
        threads::thread_init_data data(
            std::move(func), d, priority, schedulehint, stacksize);

        register_work_plain(pool, data, initial_state, ec);
    }

    inline void register_non_suspendable_work_plain(
        threads::thread_pool_base* pool, threads::thread_function_type&& func,
        util::thread_description const& description =
            util::thread_description(),
        threads::thread_schedule_state initial_state =
            threads::thread_schedule_state::pending,
        threads::thread_priority priority = threads::thread_priority::normal,
        threads::thread_schedule_hint schedulehint =
            threads::thread_schedule_hint(),
        error_code& ec = throws)
    {
        HPX_ASSERT(pool);
        register_work_plain(pool, std::move(func), description, initial_state,
            priority, schedulehint, threads::thread_stacksize::nostack, ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Create a new work item using the given function as the
    ///        work to be executed.
    ///
    /// \note This function is completely equivalent to the first overload
    ///       of threads#register_work_plain above, except that part of the
    ///       parameters are passed as members of the threads#thread_init_data
    ///       object.
    ///
    inline void register_work_plain(threads::thread_init_data& data,
        threads::thread_schedule_state initial_state =
            threads::thread_schedule_state::pending,
        error_code& ec = throws)
    {
        register_work_plain(
            detail::get_self_or_default_pool(), data, initial_state, ec);
    }

    inline void register_non_suspendable_work_plain(
        threads::thread_init_data& data,
        threads::thread_schedule_state initial_state =
            threads::thread_schedule_state::pending,
        error_code& ec = throws)
    {
        register_non_suspendable_work_plain(
            detail::get_self_or_default_pool(), data, initial_state, ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Create a new work item using the given function as the
    ///        work to be executed. This work item will be used to create a
    ///        \a threads#thread instance whenever the shepherd thread runs out
    ///        of work only. The created work descriptions will be queued
    ///        separately, causing them to be converted into actual thread
    ///        objects on a first-come-first-served basis.
    ///
    /// \param func       [in] The function to be executed as the thread-function.
    ///                   This function has to expose the minimal low level
    ///                   HPX-thread interface, i.e. it takes one argument (a
    ///                   \a threads#thread_restart_state) and returns a
    ///                   \a threads#thread_schedule_state.
    /// \param description [in] A optional string describing the newly created
    ///                   thread. This is useful for debugging and logging
    ///                   purposes as this string will be inserted in the logs.
    /// \param initial_state [in] The thread state the newly created thread
    ///                   should have. If this is not given it defaults to
    ///                   \a threads#pending, which means that the new thread
    ///                   will be scheduled to run as soon as it is created.
    /// \param priority   [in] This is the priority the newly created HPX-thread
    ///                   should be executed with. The default is \a
    ///                   threads#thread_priority::normal. This parameter is not
    ///                   guaranteed to be taken into account as it depends on
    ///                   the used scheduling policy whether priorities are
    ///                   supported in the first place.
    /// \param os_thread  [in] The number of the shepherd thread the newly
    ///                   created HPX-thread should run on. If this is given it
    ///                   will be no more than a hint in any case, mainly
    ///                   because even if the HPX-thread gets scheduled on the
    ///                   queue of the requested shepherd thread, it still can
    ///                   be stolen by another shepherd thread. If this is not
    ///                   given, the system will select a shepherd thread.
    /// \param ec         [in,out] This represents the error status on exit,
    ///                   if this is pre-initialized to \a hpx#throws
    ///                   the function will throw on error instead.
    ///
    /// \note The value returned by the thread function will be interpreted by
    ///       the thread manager as the new thread state the executed HPX-thread
    ///       needs to be switched to. Normally, HPX-threads will either return
    ///       \a threads#terminated (if the thread should be destroyed) or
    ///       \a threads#suspended (if the thread needs to be suspended because
    ///       it is waiting for an external event to happen). The external
    ///       event will set the state of the thread back to pending, which
    ///       will re-schedule the HPX-thread.
    ///
    /// \throws invalid_status if the runtime system has not been started yet.
    ///
    inline void register_work_plain(threads::thread_function_type&& func,
        util::thread_description const& description =
            util::thread_description(),
        threads::thread_schedule_state initial_state =
            threads::thread_schedule_state::pending,
        threads::thread_priority priority = threads::thread_priority::normal,
        threads::thread_schedule_hint schedulehint =
            threads::thread_schedule_hint(),
        threads::thread_stacksize stacksize =
            threads::thread_stacksize::default_,
        error_code& ec = throws)
    {
        register_work_plain(detail::get_self_or_default_pool(), std::move(func),
            description, initial_state, priority, schedulehint, stacksize, ec);
    }

    inline void register_non_suspendable_work_plain(
        threads::thread_function_type&& func,
        util::thread_description const& description =
            util::thread_description(),
        threads::thread_schedule_state initial_state =
            threads::thread_schedule_state::pending,
        threads::thread_priority priority = threads::thread_priority::normal,
        threads::thread_schedule_hint schedulehint =
            threads::thread_schedule_hint(),
        error_code& ec = throws)
    {
        register_non_suspendable_work_plain(detail::get_self_or_default_pool(),
            std::move(func), description, initial_state, priority, schedulehint,
            ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Create a new work item using the given function as the
    ///        work to be executed.
    ///
    /// \param func       [in] The function to be executed as the thread-function.
    ///                   This function has to expose the minimal low level
    ///                   HPX-thread interface, i.e. it takes one argument (a
    ///                   \a threads#thread_restart_state). The thread will be
    ///                   terminated after the function returns.
    ///
    /// \note All other arguments are equivalent to those of the function
    ///       \a threads#register_work_plain
    ///
    template <typename F>
    void register_work(F&& func,
        util::thread_description const& description =
            util::thread_description(),
        threads::thread_schedule_state initial_state =
            threads::thread_schedule_state::pending,
        threads::thread_priority priority = threads::thread_priority::normal,
        threads::thread_schedule_hint os_thread =
            threads::thread_schedule_hint(),
        threads::thread_stacksize stacksize =
            threads::thread_stacksize::default_,
        error_code& ec = throws)
    {
        threads::thread_function_type thread_func(
            detail::thread_function<typename std::decay<F>::type>{
                std::forward<F>(func)});
        register_work_plain(std::move(thread_func), description, initial_state,
            priority, os_thread, stacksize, ec);
    }

    template <typename F>
    void register_non_suspendable_work(F&& func,
        util::thread_description const& description =
            util::thread_description(),
        threads::thread_schedule_state initial_state =
            threads::thread_schedule_state::pending,
        threads::thread_priority priority = threads::thread_priority::normal,
        threads::thread_schedule_hint os_thread =
            threads::thread_schedule_hint(),
        error_code& ec = throws)
    {
        threads::thread_function_type thread_func(
            detail::thread_function<typename std::decay<F>::type>{
                std::forward<F>(func)});
        register_non_suspendable_work_plain(detail::get_self_or_default_pool(),
            std::move(thread_func), description, initial_state, priority,
            os_thread, ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Create a new work item using the given function as the
    ///        work to be executed.
    ///
    /// \param func       [in] The function to be executed as the thread-function.
    ///                   This function has to expose the minimal low level
    ///                   HPX-thread interface, i.e. it takes no arguments. The
    ///                   thread will be terminated after the function returns.
    ///
    /// \note All other arguments are equivalent to those of the function
    ///       \a threads#register_work_plain
    ///
    template <typename F>
    void register_work_nullary(F&& func,
        util::thread_description const& description =
            util::thread_description(),
        threads::thread_schedule_state initial_state =
            threads::thread_schedule_state::pending,
        threads::thread_priority priority = threads::thread_priority::normal,
        threads::thread_schedule_hint os_thread =
            threads::thread_schedule_hint(),
        threads::thread_stacksize stacksize =
            threads::thread_stacksize::default_,
        error_code& ec = throws)
    {
        threads::thread_function_type thread_func(
            detail::thread_function_nullary<typename std::decay<F>::type>{
                std::forward<F>(func)});
        register_work_plain(std::move(thread_func), description, initial_state,
            priority, os_thread, stacksize, ec);
    }

    template <typename F>
    void register_non_suspendable_work_nullary(F&& func,
        util::thread_description const& description =
            util::thread_description(),
        threads::thread_schedule_state initial_state =
            threads::thread_schedule_state::pending,
        threads::thread_priority priority = threads::thread_priority::normal,
        threads::thread_schedule_hint os_thread =
            threads::thread_schedule_hint(),
        error_code& ec = throws)
    {
        threads::thread_function_type thread_func(
            detail::thread_function_nullary<typename std::decay<F>::type>{
                std::forward<F>(func)});
        register_non_suspendable_work_plain(detail::get_self_or_default_pool(),
            std::move(thread_func), description, initial_state, priority,
            os_thread, ec);
    }

    template <typename F>
    void register_work_nullary(threads::thread_pool_base* pool, F&& func,
        util::thread_description const& description =
            util::thread_description(),
        threads::thread_schedule_state initial_state =
            threads::thread_schedule_state::pending,
        threads::thread_priority priority = threads::thread_priority::normal,
        threads::thread_schedule_hint os_thread =
            threads::thread_schedule_hint(),
        threads::thread_stacksize stacksize =
            threads::thread_stacksize::default_,
        error_code& ec = throws)
    {
        threads::thread_function_type thread_func(
            detail::thread_function_nullary<typename std::decay<F>::type>{
                std::forward<F>(func)});
        register_work_plain(pool, std::move(thread_func), description,
            initial_state, priority, os_thread, stacksize, ec);
    }

    template <typename F>
    void register_non_suspendable_work_nullary(threads::thread_pool_base* pool,
        F&& func,
        util::thread_description const& description =
            util::thread_description(),
        threads::thread_schedule_state initial_state =
            threads::thread_schedule_state::pending,
        threads::thread_priority priority = threads::thread_priority::normal,
        threads::thread_schedule_hint os_thread =
            threads::thread_schedule_hint(),
        error_code& ec = throws)
    {
        threads::thread_function_type thread_func(
            detail::thread_function_nullary<typename std::decay<F>::type>{
                std::forward<F>(func)});
        register_non_suspendable_work_plain(pool, std::move(thread_func),
            description, initial_state, priority, os_thread, ec);
    }
#endif
}}    // namespace hpx::threads

///////////////////////////////////////////////////////////////////////////////
#if defined(HPX_HAVE_REGISTER_THREAD_COMPATIBILITY) &&                         \
    defined(HPX_HAVE_REGISTER_THREAD_OVERLOADS_COMPATIBILITY)
namespace hpx { namespace applier {
    using threads::register_thread;
    using threads::register_thread_nullary;
    using threads::register_thread_plain;

    using threads::register_work;
    using threads::register_work_nullary;
    using threads::register_work_plain;

    using threads::register_non_suspendable_thread;
    using threads::register_non_suspendable_thread_nullary;
    using threads::register_non_suspendable_thread_plain;

    using threads::register_non_suspendable_work;
    using threads::register_non_suspendable_work_nullary;
    using threads::register_non_suspendable_work_plain;
}}    // namespace hpx::applier
#endif

/// \endcond
