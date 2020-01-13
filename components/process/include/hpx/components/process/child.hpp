// Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PROCESS_CHILD_MAR_11_2016_0646PM)
#define HPX_PROCESS_CHILD_MAR_11_2016_0646PM

#include <hpx/config.hpp>
#include <hpx/async.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/runtime/components/client_base.hpp>
#include <hpx/runtime/launch_policy.hpp>

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
            typedef server::child::terminate_action terminate_action;
            return hpx::async(terminate_action(), this->get_id());
        }

        void terminate(launch::sync_policy)
        {
            return terminate().get();
        }

        hpx::future<int> wait_for_exit()
        {
            typedef server::child::wait_for_exit_action wait_for_exit_action;
            return hpx::async(wait_for_exit_action(), this->get_id());
        }

        int wait_for_exit(launch::sync_policy)
        {
            return wait_for_exit().get();
        }
    };
}}}

#endif
