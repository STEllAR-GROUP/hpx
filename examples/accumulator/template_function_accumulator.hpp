//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_EXAMPLES_TEMPLATE_FUNCTION_ACCUMULATOR_JUL_12_2012_1056AM)
#define HPX_EXAMPLES_TEMPLATE_FUNCTION_ACCUMULATOR_JUL_12_2012_1056AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/include/components.hpp>

#include "stubs/template_function_accumulator.hpp"

namespace examples
{
    ///////////////////////////////////////////////////////////////////////////
    /// Client for the \a server::managed_accumulator component.
    class template_function_accumulator
      : public hpx::components::client_base<
            template_function_accumulator, stubs::managed_accumulator
        >
    {
        typedef hpx::components::client_base<
            template_function_accumulator, stubs::managed_accumulator
        > base_type;

    public:
        /// Default construct an empty client side representation (not
        /// connected to any existing component).
        template_function_accumulator()
        {}

        /// Create a client side representation for the existing
        /// \a server::managed_accumulator instance with the given GID.
        template_function_accumulator(hpx::future<hpx::naming::id_type> && gid)
          : base_type(std::move(gid))
        {}

        ///////////////////////////////////////////////////////////////////////
        /// Reset the accumulator's value to 0.
        ///
        /// \note This function has fire-and-forget semantics. It will not wait
        ///       for the action to be executed. Instead, it will return
        ///       immediately after the action has has been dispatched.
        void reset_non_blocking()
        {
            HPX_ASSERT(this->get_id());
            this->base_type::reset_non_blocking(this->get_id());
        }

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
        template <typename T>
        void add_non_blocking(T arg)
        {
            HPX_ASSERT(this->get_id());
            this->base_type::add_non_blocking(this->get_id(), arg);
        }

        /// Add \p arg to the accumulator's value.
        ///
        /// \note This function is fully synchronous.
        template <typename T>
        void add_sync(T arg)
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
        hpx::lcos::future<double> query_async()
        {
            HPX_ASSERT(this->get_id());
            return this->base_type::query_async(this->get_id());
        }

        /// Query the current value of the accumulator.
        ///
        /// \note This function is fully synchronous.
        double query_sync()
        {
            HPX_ASSERT(this->get_id());
            return this->base_type::query_sync(this->get_id());
        }
    };
}

#endif

