//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TEMPLATE_ACCUMULATOR_MAR_31_2016_1052AM)
#define HPX_TEMPLATE_ACCUMULATOR_MAR_31_2016_1052AM

#include <hpx/include/components.hpp>

#include "server/template_accumulator.hpp"

#include <utility>

namespace examples
{
    ///////////////////////////////////////////////////////////////////////////
    /// Client for the \a server::accumulator component.
    template <typename T>
    class template_accumulator
      : public hpx::components::client_base<
            template_accumulator<T>, server::template_accumulator<T>
        >
    {
        typedef hpx::components::client_base<
            template_accumulator<T>, server::template_accumulator<T>
        > base_type;

        typedef typename server::template_accumulator<T>::argument_type
            argument_type;

    public:
        /// Default construct an empty client side representation (not
        /// connected to any existing component).
        template_accumulator()
        {}

        /// Create a client side representation for the existing
        /// \a server::accumulator instance with the given GID.
        template_accumulator(hpx::future<hpx::id_type> && gid)
          : base_type(std::move(gid))
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

            typedef typename server::template_accumulator<T>::reset_action
                action_type;
            hpx::apply<action_type>(this->get_id());
        }

        /// Reset the accumulator's value to 0.
        ///
        /// \note This function is fully synchronous.
        void reset()
        {
            HPX_ASSERT(this->get_id());

            typedef typename server::template_accumulator<T>::reset_action
                action_type;
            hpx::async<action_type>(hpx::launch::sync, this->get_id()).get();
        }

        ///////////////////////////////////////////////////////////////////////
        /// Add \p arg to the accumulator's value.
        ///
        /// \note This function has fire-and-forget semantics. It will not wait
        ///       for the action to be executed. Instead, it will return
        ///       immediately after the action has has been dispatched.
        void add(hpx::launch::apply_policy, argument_type arg)
        {
            HPX_ASSERT(this->get_id());

            typedef typename server::template_accumulator<T>::add_action
                action_type;
            hpx::apply<action_type>(hpx::launch::sync, this->get_id(), arg);
        }

        /// Add \p arg to the accumulator's value.
        ///
        /// \note This function is fully synchronous.
        void add(argument_type arg)
        {
            HPX_ASSERT(this->get_id());

            typedef typename server::template_accumulator<T>::add_action
                action_type;
            hpx::async<action_type>(this->get_id(), arg).get();
        }

        ///////////////////////////////////////////////////////////////////////
        /// Asynchronously query the current value of the accumulator.
        ///
        /// \returns This function returns an \a hpx::lcos::future. When the
        ///          value of this computation is needed, the get() method of
        ///          the future should be called. If the value is available,
        ///          get() will return immediately; otherwise, it will block
        ///          until the value is ready.
        hpx::future<argument_type> query(hpx::launch::async_policy)
        {
            HPX_ASSERT(this->get_id());

            typedef typename server::template_accumulator<T>::query_action
                action_type;
            return hpx::async<action_type>(hpx::launch::async, this->get_id());
        }

        /// Query the current value of the accumulator.
        ///
        /// \note This function is fully synchronous.
        argument_type query(hpx::launch::sync_policy = hpx::launch::sync)
        {
            HPX_ASSERT(this->get_id());

            typedef typename server::template_accumulator<T>::query_action
                action_type;
            return action_type()(hpx::launch::sync, this->get_id());
        }
    };
}

#endif

