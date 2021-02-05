//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx.hpp>
#include <hpx/future.hpp>
#include <hpx/modules/actions_base.hpp>
#include <hpx/include/client.hpp>

#include "stubs/throttle.hpp"

#include <cstddef>
#include <utility>

namespace throttle
{
    ///////////////////////////////////////////////////////////////////////////
    class throttle
      : public hpx::components::client_base<throttle, stubs::throttle>
    {
    private:
        typedef hpx::components::client_base<throttle, stubs::throttle> base_type;

    public:
        // create a new partition instance and initialize it synchronously
        throttle()
          : base_type(hpx::new_<server::throttle>(hpx::find_here()))
        {}

        explicit throttle(hpx::id_type const& id)
          : base_type(id)
        {}

        throttle(hpx::future<hpx::naming::id_type> && gid)
          : base_type(std::move(gid))
        {}

        ~throttle() = default;

        void suspend(std::size_t thread_num) const
        {
            return stubs::throttle::suspend(this->get_id(), thread_num);
        }

        void resume(std::size_t thread_num) const
        {
            return stubs::throttle::resume(this->get_id(), thread_num);
        }
    };
}

#endif
