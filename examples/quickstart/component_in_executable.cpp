////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2012 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#include <hpx/hpx_main.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/actions.hpp>

using hpx::components::client_base;
using hpx::components::simple_component;
using hpx::components::simple_component_base;
using hpx::components::new_;

using hpx::find_here;

struct hello_world_server : simple_component_base<hello_world_server>
{
};

typedef simple_component<hello_world_server> server_type;
HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(server_type, hello_world_server);

struct hello_world : client_base<hello_world, hello_world_server>
{
    typedef client_base<hello_world, hello_world_server> base_type;

    hello_world(hpx::future<hpx::id_type> && id) : base_type(std::move(id)) {}
};

///////////////////////////////////////////////////////////////////////////////
int main()
{
    hello_world hw = new_<hello_world_server>(find_here());

    return 0;
}

