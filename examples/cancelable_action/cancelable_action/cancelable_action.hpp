//  Copyright (c) 2007-2011 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/assert.hpp>
#include <hpx/include/components.hpp>

#include "stubs/cancelable_action.hpp"

#include <utility>

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
        cancelable_action() = default;

        /// Create a client side representation of an object which is newly
        /// created on the given locality
        explicit cancelable_action(hpx::naming::id_type const& target_gid)
          : base_type(hpx::new_<cancelable_action>(target_gid))
        {}

        cancelable_action(hpx::future<hpx::id_type>&& target_gid)
          : base_type(std::move(target_gid))
        {}

        ///////////////////////////////////////////////////////////////////////
        void do_it(hpx::error_code& ec = hpx::throws)
        {
            HPX_ASSERT(this->get_id());
            this->base_type::do_it(this->get_id(), ec);
        }

        void cancel_it()
        {
            HPX_ASSERT(this->get_id());
            this->base_type::cancel_it(this->get_id());
        }
    };
}


#endif
