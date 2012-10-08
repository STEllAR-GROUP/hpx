//  Copyright (c) 2007-2011 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_EXAMPLE_CANCELABLE_ACTION_APR_19_1052AM)
#define HPX_EXAMPLE_CANCELABLE_ACTION_APR_19_1052AM

#include <hpx/include/components.hpp>

#include "stubs/cancelable_action.hpp"

namespace examples
{
    ///////////////////////////////////////////////////////////////////////////
    // Client side representation for for the \a server::cancelable_action
    // component.
    class cancelable_action
      : public hpx::components::client_base<
            cancelable_action, stubs::cancelable_action
        >
    {
        typedef hpx::components::client_base<
            cancelable_action, stubs::cancelable_action
        > base_type;

    public:
        // Default construct an empty client side representation (not
        // connected to any existing component).
        cancelable_action()
        {}

        /// Create a client side representation of an object which is newly
        /// created on the given locality
        cancelable_action(hpx::naming::id_type const& target_gid)
          : base_type(base_type::stub_type::create(target_gid))
        {}

        ///////////////////////////////////////////////////////////////////////
        void do_it(hpx::error_code& ec = hpx::throws)
        {
            BOOST_ASSERT(this->gid_);
            this->base_type::do_it(this->gid_, ec);
        }

        void cancel_it()
        {
            BOOST_ASSERT(this->gid_);
            this->base_type::cancel_it(this->gid_);
        }
    };
}

#endif

