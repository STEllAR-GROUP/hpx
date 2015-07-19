//  Copyright (c) 2007-2011 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_F976C441_3AEF_4CC3_A47C_A51042E0F12C)
#define HPX_F976C441_3AEF_4CC3_A47C_A51042E0F12C

#include <hpx/include/components.hpp>

#include "stubs/managed_accumulator.hpp"

namespace examples
{
    ///////////////////////////////////////////////////////////////////////////
    /// Client for the \a server::managed_accumulator component.
    //[managed_accumulator_client_inherit
    class managed_accumulator
      : public hpx::components::client_base<
            managed_accumulator, stubs::managed_accumulator
        >
    //]
    {
        //[managed_accumulator_base_type
        typedef hpx::components::client_base<
            managed_accumulator, stubs::managed_accumulator
        > base_type;
        //]

        typedef base_type::argument_type argument_type;

    public:
        /// Default construct an empty client side representation (not
        /// connected to any existing component).
        managed_accumulator()
        {}

        /// Create a client side representation for the existing
        /// \a server::managed_accumulator instance with the given GID.
        managed_accumulator(hpx::future<hpx::naming::id_type> && gid)
          : base_type(std::move(gid))
        {}

        ///////////////////////////////////////////////////////////////////////
        /// Reset the accumulator's value to 0.
        ///
        /// \note This function has fire-and-forget semantics. It will not wait
        ///       for the action to be executed. Instead, it will return
        ///       immediately after the action has has been dispatched.
        //[managed_accumulator_client_reset_non_blocking
        void reset_non_blocking()
        {
            HPX_ASSERT(this->get_id());
            this->base_type::reset_non_blocking(this->get_id());
        }
        //]

        /// Reset the accumulator's value to 0.
        ///
        /// \note This function is fully synchronous.
        void reset_sync()
        {
            HPX_ASSERT(this->get_id());
            this->base_type::reset_sync(this->get_id());
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
            this->base_type::add_non_blocking(this->get_id(), arg);
        }

        /// Add \p arg to the accumulator's value.
        ///
        /// \note This function is fully synchronous.
        //[managed_accumulator_client_add_sync
        void add_sync(argument_type arg)
        {
            HPX_ASSERT(this->get_id());
            this->base_type::add_sync(this->get_id(), arg);
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
        //[managed_accumulator_client_query_async
        hpx::lcos::future<argument_type> query_async()
        {
            HPX_ASSERT(this->get_id());
            return this->base_type::query_async(this->get_id());
        }
        //]

        /// Query the current value of the accumulator.
        ///
        /// \note This function is fully synchronous.
        argument_type query_sync()
        {
            HPX_ASSERT(this->get_id());
            return this->base_type::query_sync(this->get_id());
        }
    };
}

#endif

