//  Copyright (c) 2007-2015 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_ECFE19F9_A826_4AE1_AC7C_33DC5714CF0B)
#define HPX_ECFE19F9_A826_4AE1_AC7C_33DC5714CF0B

#include <hpx/include/components.hpp>

#include "server/accumulator.hpp"

namespace examples
{
    ///////////////////////////////////////////////////////////////////////////
    /// Client for the \a server::accumulator component.
    //[accumulator_client_inherit
    class accumulator
      : public hpx::components::client_base<
            accumulator, server::accumulator
        >
    //]
    {
        //[accumulator_base_type
        typedef hpx::components::client_base<
            accumulator, server::accumulator
        > base_type;
        //]

        typedef server::accumulator::argument_type argument_type;

    public:
        /// Default construct an empty client side representation (not
        /// connected to any existing component).
        accumulator()
        {}

        /// Create a client side representation for the existing
        /// \a server::accumulator instance with the given GID.
        accumulator(hpx::future<hpx::id_type> && gid)
          : base_type(std::move(gid))
        {}

        ///////////////////////////////////////////////////////////////////////
        /// Reset the accumulator's value to 0.
        ///
        /// \note This function has fire-and-forget semantics. It will not wait
        ///       for the action to be executed. Instead, it will return
        ///       immediately after the action has has been dispatched.
        //[accumulator_client_reset_non_blocking
        void reset_non_blocking()
        {
            HPX_ASSERT(this->get_id());

            typedef server::accumulator::reset_action action_type;
            hpx::apply<action_type>(this->get_id());
        }
        //]

        /// Reset the accumulator's value to 0.
        ///
        /// \note This function is fully synchronous.
        void reset_sync()
        {
            HPX_ASSERT(this->get_id());

            typedef server::accumulator::reset_action action_type;
            hpx::async<action_type>(this->get_id()).get();
        }

        ///////////////////////////////////////////////////////////////////////
        /// Add \p arg to the accumulator's value.
        ///
        /// \note This function has fire-and-forget semantics. It will not wait
        ///       for the action to be executed. Instead, it will return
        ///       immediately after the action has has been dispatched.
        void add_non_blocking(argument_type arg)
        {
            HPX_ASSERT(this->get_id());

            typedef server::accumulator::add_action action_type;
            hpx::apply<action_type>(this->get_id(), arg);
        }

        /// Add \p arg to the accumulator's value.
        ///
        /// \note This function is fully synchronous.
        //[accumulator_client_add_sync
        void add_sync(argument_type arg)
        {
            HPX_ASSERT(this->get_id());

            typedef server::accumulator::add_action action_type;
            hpx::async<action_type>(this->get_id(), arg).get();
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
        //[accumulator_client_query_async
        hpx::future<argument_type> query_async()
        {
            HPX_ASSERT(this->get_id());

            typedef server::accumulator::query_action action_type;
            return hpx::async<action_type>(this->get_id());
        }
        //]

        /// Query the current value of the accumulator.
        ///
        /// \note This function is fully synchronous.
        argument_type query_sync()
        {
            HPX_ASSERT(this->get_id());

            return query_async().get();
        }
    };
}

#endif

