//  Copyright (c) 2012 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//[hello_world_cpp_getting_started
#include "hello_world_component.hpp"
#include <hpx/include/iostreams.hpp>

namespace examples { namespace server
{

void hello_world::invoke()
{
    hpx::cout << "Hello HPX World!\n" << hpx::flush;
}

}}

HPX_REGISTER_COMPONENT_MODULE();

typedef hpx::components::component<
    examples::server::hello_world
> hello_world_type;

HPX_REGISTER_COMPONENT(hello_world_type, hello_world);

HPX_REGISTER_ACTION(
    examples::server::hello_world::invoke_action, hello_world_invoke_action);
//]

