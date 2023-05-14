//  Copyright (c) 2007-2023 Hartmut Kaiser
//  Copyright (c)      2018 Thomas Heller
//  Copyright (c)      2011 Bryce Lelbach
//  Copyright (c) 2008-2009 Chirag Dekate, Anshul Tandon
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/threading_base/thread_init_data.hpp>
#include <hpx/threading_base/threading_base_fwd.hpp>

#include <type_traits>
#include <utility>

namespace hpx::threads {

    namespace detail {

        HPX_CORE_EXPORT threads::thread_result_type cleanup_thread();

        template <typename F>
        struct thread_function
        {
            F f;

            inline threads::thread_result_type operator()(
                threads::thread_arg_type)
            {
                // execute the actual thread function
                f(threads::thread_restart_state::signaled);

                return cleanup_thread();
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

                return cleanup_thread();
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
    /// \param id         [out] The id of the newly created thread (if applicable)
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
    HPX_CORE_EXPORT void register_thread(threads::thread_init_data& data,
        threads::thread_pool_base* pool, threads::thread_id_ref_type& id,
        error_code& ec = hpx::throws);

    HPX_CORE_EXPORT threads::thread_id_ref_type register_thread(
        threads::thread_init_data& data, threads::thread_pool_base* pool,
        error_code& ec = hpx::throws);

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Create a new \a thread using the given data on the same thread
    ///        pool as the calling thread, or on the default thread pool if not
    ///        on an HPX thread.
    ///
    /// \param data       [in] The data to use for creating the thread.
    /// \param id         [out] The id of the newly created thread (if applicable)
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
    HPX_CORE_EXPORT void register_thread(threads::thread_init_data& data,
        threads::thread_id_ref_type& id, error_code& ec = throws);

    HPX_CORE_EXPORT threads::thread_id_ref_type register_thread(
        threads::thread_init_data& data, error_code& ec = throws);

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
    HPX_CORE_EXPORT thread_id_ref_type register_work(
        threads::thread_init_data& data, threads::thread_pool_base* pool,
        error_code& ec = hpx::throws);

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
    HPX_CORE_EXPORT thread_id_ref_type register_work(
        threads::thread_init_data& data, error_code& ec = throws);
}    // namespace hpx::threads

/// \endcond
