//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_EXAMPLES_STUBS_TEMPLATE_FUNCTION_ACCUMULATOR_JUL_12_2012_1056AM)
#define HPX_EXAMPLES_STUBS_TEMPLATE_FUNCTION_ACCUMULATOR_JUL_12_2012_1056AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/applier.hpp>
#include <hpx/include/async.hpp>

#include "../server/template_function_accumulator.hpp"

namespace examples { namespace stubs
{
    ///////////////////////////////////////////////////////////////////////////
    struct managed_accumulator
      : hpx::components::stub_base<server::template_function_accumulator>
    {
        ///////////////////////////////////////////////////////////////////////
        /// Reset the accumulator's value to 0.
        ///
        /// \note This function has fire-and-forget semantics. It will not wait
        ///       for the action to be executed. Instead, it will return
        ///       immediately after the action has has been dispatched.
        static void reset_non_blocking(hpx::naming::id_type const& gid)
        {
            typedef server::template_function_accumulator::reset_action action_type;
            hpx::apply<action_type>(gid);
        }

        /// Reset the accumulator's value to 0.
        ///
        /// \note This function is fully synchronous.
        static void reset_sync(hpx::naming::id_type const& gid)
        {
            typedef server::template_function_accumulator::reset_action action_type;
            hpx::async<action_type>(gid).get();
        }

        ///////////////////////////////////////////////////////////////////////
        /// Add \p arg to the accumulator's value.
        ///
        /// \note This function has fire-and-forget semantics. It will not wait
        ///       for the action to be executed. Instead, it will return
        ///       immediately after the action has has been dispatched.
        template <typename T>
        static void
        add_non_blocking(hpx::naming::id_type const& gid, T arg)
        {
            typedef server::template_function_accumulator::add_action<T> action_type;
            hpx::apply<action_type>(gid, arg);
        }

        /// Add \p arg to the accumulator's value.
        ///
        /// \note This function is fully synchronous.
        //[managed_accumulator_stubs_add_sync
        template <typename T>
        static void
        add_sync(hpx::naming::id_type const& gid, T arg)
        {
            typedef typename server::template_function_accumulator::add_action<T>
                 action_type;
            hpx::async<action_type>(gid, arg).get();
        }

        ///////////////////////////////////////////////////////////////////////
        /// Asynchronously query the current value of the accumulator.
        ///
        /// \returns This function returns an \a hpx::lcos::future. When the
        ///          value of this computation is needed, the get() method of
        ///          the future should be called. If the value is available,
        ///          get() will return immediately; otherwise, it will block
        ///          until the value is ready.
        static hpx::lcos::future<double>
        query_async(hpx::naming::id_type const& gid)
        {
            typedef server::template_function_accumulator::query_action action_type;
            return hpx::async<action_type>(gid);
        }

        /// Query the current value of the accumulator.
        ///
        /// \note This function is fully synchronous.
        static double query_sync(hpx::naming::id_type const& gid)
        {
            // The following get yields control while the action is executed.
            return query_async(gid).get();
        }
    };
}}

#endif

