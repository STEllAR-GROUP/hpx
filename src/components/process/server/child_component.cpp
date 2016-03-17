//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/runtime/threads/run_as_os_thread.hpp>

#include <hpx/components/process/child.hpp>
#include <hpx/components/process/server/child.hpp>

#include <hpx/components/process/util/terminate.hpp>
#include <hpx/components/process/util/wait_for_exit.hpp>

///////////////////////////////////////////////////////////////////////////////
HPX_REGISTER_ACTION(
    hpx::components::process::server::child::terminate_action,
    hpx_components_process_server_child_terminate_action);

HPX_REGISTER_ACTION(
    hpx::components::process::server::child::wait_for_exit_action,
    hpx_components_process_server_child_wait_for_exit);

namespace hpx { namespace components { namespace process { namespace server
{
    void child::terminate()
    {
        process::util::terminate(child_);
    }

    int child::wait_for_exit()
    {
        int (*f)(process::util::child const&) =
            &process::util::wait_for_exit<process::util::child>;
        return hpx::threads::run_as_os_thread(f, std::ref(child_)).get();
    }
}}}}

