//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(LAUNCH_PROCESS_TEST_SERVER)
#define LAUNCH_PROCESS_TEST_SERVER

#include <hpx/include/actions.hpp>
#include <hpx/include/components.hpp>

#include <string>

namespace launch_process
{
    // Test-component which is instantiated by the launching process and
    // registered with AGAS such that the launched process can use it.
    struct test_server
      : hpx::components::component_base<test_server>
    {
        test_server()
          : msg_("initialized")
        {}

        std::string get_message() const
        {
            return msg_;
        }
        HPX_DEFINE_COMPONENT_ACTION(test_server, get_message,
            get_message_action);

        void set_message(std::string const& msg)
        {
            msg_ = msg;
        }
        HPX_DEFINE_COMPONENT_ACTION(test_server, set_message,
            set_message_action);

        std::string msg_;
    };
}

typedef launch_process::test_server::get_message_action
    launch_process_get_message_action;
typedef launch_process::test_server::set_message_action
    launch_process_set_message_action;

HPX_REGISTER_ACTION_DECLARATION(launch_process_get_message_action);
HPX_REGISTER_ACTION_DECLARATION(launch_process_set_message_action);

#endif
