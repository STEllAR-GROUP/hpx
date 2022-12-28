//  Copyright (c) 2007-2022 Hartmut Kaiser
//  Copyright (c)      2018 Thomas Heller
//  Copyright (c)      2011 Bryce Lelbach
//  Copyright (c) 2008-2009 Chirag Dekate, Anshul Tandon
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/functional/function.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/threading_base/detail/get_default_pool.hpp>
#include <hpx/threading_base/scheduler_base.hpp>
#include <hpx/threading_base/thread_data.hpp>
#include <hpx/threading_base/thread_pool_base.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace hpx::threads {

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Create a new \a thread using the given function as the work to be
    ///        executed.
    ///
    /// \param func       [in] The function to be executed as the
    ///                   thread-function. This function has to expose the
    ///                   minimal low level HPX-thread interface, i.e. it takes
    ///                   one argument (a \a threads#thread_restart_state). The
    ///                   thread will be terminated after the function returns.
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
                // OS-thread. This will throw if there are still any locks held.
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
                // OS-thread. This will throw if there are still any locks held.
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
        return {detail::thread_function<std::decay_t<F>>{HPX_FORWARD(F, f)}};
    }

    template <typename F>
    thread_function_type make_thread_function_nullary(F&& f)
    {
        return {detail::thread_function_nullary<std::decay_t<F>>{
            HPX_FORWARD(F, f)}};
    }

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Create a new \a thread using the given data.
    ///
    /// \param data       [in] The data to use for creating the thread.
    /// \param pool       [in] The thread pool to use for launching the work.
    /// \param ec         [in,out] This represents the error status on exit,
    ///                   if this is pre-initialized to \a hpx#throws the
    ///                   function will throw on error instead.
    ///
    /// \returns This function will return the internal id of the newly created
    ///          HPX-thread.
    ///
    /// \throws invalid_status if the runtime system has not been started yet.
    ///
    /// \note             As long as \a ec is not pre-initialized to
    ///                   \a hpx#throws this function doesn't
    ///                   throw but returns the result code using the parameter
    ///                   \a ec. Otherwise it throws an instance
    ///                   of hpx#exception.
    inline threads::thread_id_ref_type register_thread(
        threads::thread_init_data& data, threads::thread_pool_base* pool,
        error_code& ec = throws)
    {
        HPX_ASSERT(pool);
        data.run_now = true;
        threads::thread_id_ref_type id = threads::invalid_thread_id;
        pool->create_thread(data, id, ec);
        return id;
    }

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Create a new \a thread using the given data on the same thread
    ///        pool as the calling thread, or on the default thread pool if not
    ///        on an HPX thread.
    ///
    /// \param data       [in] The data to use for creating the thread.
    /// \param ec         [in,out] This represents the error status on exit,
    ///                   if this is pre-initialized to \a hpx#throws the
    ///                   function will throw on error instead.
    ///
    /// \returns This function will return the internal id of the newly created
    ///          HPX-thread.
    ///
    /// \throws invalid_status if the runtime system has not been started yet.
    ///
    /// \note             As long as \a ec is not pre-initialized to
    ///                   \a hpx#throws this function doesn't throw but returns
    ///                   the result code using the parameter \a ec. Otherwise
    ///                   it throws an instance of hpx#exception.
    inline threads::thread_id_ref_type register_thread(
        threads::thread_init_data& data, error_code& ec = throws)
    {
        return register_thread(data, detail::get_self_or_default_pool(), ec);
    }

    /// \brief Create a new work item using the given data.
    ///
    /// \param data       [in] The data to use for creating the thread.
    /// \param pool       [in] The thread pool to use for launching the work.
    /// \param ec         [in,out] This represents the error status on exit,
    ///                   if this is pre-initialized to \a hpx#throws the
    ///                   function will throw on error instead.
    ///
    /// \throws invalid_status if the runtime system has not been started yet.
    ///
    /// \note             As long as \a ec is not pre-initialized to
    ///                   \a hpx#throws this function doesn't throw but returns
    ///                   the result code using the parameter \a ec. Otherwise
    ///                   it throws an instance of hpx#exception.
    inline thread_id_ref_type register_work(threads::thread_init_data& data,
        threads::thread_pool_base* pool, error_code& ec = throws)
    {
        HPX_ASSERT(pool);
        data.run_now = false;
        return pool->create_work(data, ec);
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
    inline thread_id_ref_type register_work(
        threads::thread_init_data& data, error_code& ec = throws)
    {
        return register_work(data, detail::get_self_or_default_pool(), ec);
    }
}    // namespace hpx::threads

/// \endcond
