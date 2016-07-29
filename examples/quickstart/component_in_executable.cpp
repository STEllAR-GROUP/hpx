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

#include <utility>

using hpx::components::stub_base;
using hpx::components::client_base;
using hpx::components::component;
using hpx::components::component_base;

using hpx::find_here;
using hpx::async;

using hpx::cout;
using hpx::flush;

struct hello_world_server : component_base<hello_world_server>
{
    void print() const { cout << "hello world\n" << flush; }

    HPX_DEFINE_COMPONENT_ACTION(hello_world_server, print, print_action);
};

typedef component<hello_world_server> server_type;
HPX_REGISTER_COMPONENT(server_type, hello_world_server);

typedef hello_world_server::print_action print_action;
HPX_REGISTER_ACTION_DECLARATION(print_action);
HPX_REGISTER_ACTION(print_action);

struct hello_world : client_base<hello_world, hello_world_server>
{
    typedef client_base<hello_world, hello_world_server> base_type;

    hello_world(hpx::future<hpx::id_type> && id) : base_type(std::move(id)) {}

    void print() { async<print_action>(this->get_id()).get(); }
};

///////////////////////////////////////////////////////////////////////////////
int main()
{
    hello_world hw = hello_world::create(find_here());

    hw.print();

    return 0;
}

