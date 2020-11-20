// Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <hpx/assert.hpp>
#include <hpx/async_base/launch_policy.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/modules/async_distributed.hpp>
#include <hpx/runtime/components/client_base.hpp>

#include <hpx/components/process/server/child.hpp>

#include <utility>

namespace hpx { namespace components { namespace process
{
    ///////////////////////////////////////////////////////////////////////////
    class child : public client_base<child, process::server::child>
    {
        typedef client_base<child, process::server::child> base_type;

    public:
        template <typename ... Ts>
        child(Ts &&... ts)
          : base_type(std::forward<Ts>(ts)...)
        {}

        hpx::future<void> terminate()
        {
#if !defined(HPX_COMPUTE_DEVICE_CODE)
            typedef server::child::terminate_action terminate_action;
            return hpx::async(terminate_action(), this->get_id());
#else
            HPX_ASSERT(false);
            return hpx::make_ready_future();
#endif
        }

        void terminate(launch::sync_policy)
        {
            return terminate().get();
        }

        hpx::future<int> wait_for_exit()
        {
#if !defined(HPX_COMPUTE_DEVICE_CODE)
            typedef server::child::wait_for_exit_action wait_for_exit_action;
            return hpx::async(wait_for_exit_action(), this->get_id());
#else
            HPX_ASSERT(false);
            return hpx::make_ready_future(int{});
#endif
        }

        int wait_for_exit(launch::sync_policy)
        {
            return wait_for_exit().get();
        }
    };
}}}

