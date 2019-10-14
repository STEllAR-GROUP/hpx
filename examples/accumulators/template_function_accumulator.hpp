//  Copyright (c) 2007-2015 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Adelstein-Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_EXAMPLES_TEMPLATE_FUNCTION_ACCUMULATOR_JUL_12_2012_1056AM)
#define HPX_EXAMPLES_TEMPLATE_FUNCTION_ACCUMULATOR_JUL_12_2012_1056AM

#include <hpx/hpx.hpp>
#include <hpx/include/components.hpp>

#include "server/template_function_accumulator.hpp"

#include <utility>

namespace examples
{
    ///////////////////////////////////////////////////////////////////////////
    /// Client for the \a server::managed_accumulator component.
    class template_function_accumulator
      : public hpx::components::client_base<
            template_function_accumulator,
            server::template_function_accumulator
        >
    {
        typedef hpx::components::client_base<
            template_function_accumulator,
            server::template_function_accumulator
        > base_type;

    public:
        /// Default construct an empty client side representation (not
        /// connected to any existing component).
        template_function_accumulator()
        {}

        /// Create a client side representation for the existing
        /// \a server::managed_accumulator instance with the given GID.
        template_function_accumulator(hpx::future<hpx::id_type> && id)
          : base_type(std::move(id))
        {}

        template_function_accumulator(hpx::id_type && id)
          : base_type(std::move(id))
        {}

        ///////////////////////////////////////////////////////////////////////
        /// Reset the accumulator's value to 0.
        ///
        /// \note This function has fire-and-forget semantics. It will not wait
        ///       for the action to be executed. Instead, it will return
        ///       immediately after the action has has been dispatched.
        void reset(hpx::launch::apply_policy)
        {
            HPX_ASSERT(this->get_id());

            typedef server::template_function_accumulator::reset_action
                action_type;
            hpx::apply<action_type>(this->get_id());
        }

        /// Reset the accumulator's value to 0.
        ///
        /// \note This function is fully synchronous.
        void reset()
        {
            HPX_ASSERT(this->get_id());

            typedef server::template_function_accumulator::reset_action
                action_type;
            action_type()(this->get_id());
        }

        ///////////////////////////////////////////////////////////////////////
        /// Add \p arg to the accumulator's value.
        ///
        /// \note This function has fire-and-forget semantics. It will not wait
        ///       for the action to be executed. Instead, it will return
        ///       immediately after the action has has been dispatched.
        template <typename T>
        void add(hpx::launch::apply_policy, T arg)
        {
            HPX_ASSERT(this->get_id());

            typedef server::template_function_accumulator::add_action<T>
                action_type;
            hpx::apply<action_type>(this->get_id(), arg);
        }

        /// Add \p arg to the accumulator's value.
        ///
        /// \note This function is fully synchronous.
        template <typename T>
        void add(T arg)
        {
            HPX_ASSERT(this->get_id());

            typedef typename server::template_function_accumulator::add_action<T>
                 action_type;
            action_type()(this->get_id(), arg);
        }
        //]

        ///////////////////////////////////////////////////////////////////////
        /// Asynchronously query the current value of the accumulator.
        ///
        /// \returns This function returns an \a hpx::lcos::future. When the
        ///          value of this computation is needed, the get() method of
        ///          the future should be called. If the value is available,
        ///          get() will return immediately; otherwise, it will block
        ///          until the value is ready.
        hpx::future<double> query(hpx::launch::async_policy)
        {
            HPX_ASSERT(this->get_id());

            typedef server::template_function_accumulator::query_action
                action_type;
            return hpx::async<action_type>(hpx::launch::async, this->get_id());
        }

        /// Query the current value of the accumulator.
        ///
        /// \note This function is fully synchronous.
        double query(hpx::launch::sync_policy = hpx::launch::sync)
        {
            HPX_ASSERT(this->get_id());

            typedef server::template_function_accumulator::query_action
                action_type;
            return action_type()(this->get_id());
        }
    };
}

#endif

