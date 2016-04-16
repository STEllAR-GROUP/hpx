////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2012 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#include <hpx/hpx_main.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/iostreams.hpp>

#include <string>

using hpx::components::stub_base;
using hpx::components::client_base;
using hpx::components::component;
using hpx::components::component_base;

using hpx::id_type;
using hpx::find_here;
using hpx::async;

using hpx::cout;
using hpx::flush;

struct message_server : component_base<message_server>
{
    std::string msg;

    message_server() : msg("uninitialized\n") {}

    message_server(std::string const& msg_) : msg(msg_) {}

    void print() const { cout << msg << flush; }

    HPX_DEFINE_COMPONENT_ACTION(message_server, print, print_action);
};

typedef component<message_server> server_type;
HPX_REGISTER_COMPONENT(server_type, message_server);

typedef message_server::print_action print_action;
HPX_REGISTER_ACTION_DECLARATION(print_action);
HPX_REGISTER_ACTION(print_action);

struct message : client_base<message, message_server>
{
    typedef client_base<message, message_server> base_type;

    message(hpx::future<hpx::id_type> && id) : base_type(std::move(id)) {}

    void print() { async<print_action>(this->get_id()).get(); }
};

///////////////////////////////////////////////////////////////////////////////
int main()
{
    std::string msg = "hello world\n";
    message ms = message::create(hpx::find_here(), msg);

    ms.print();

    return 0;
}

