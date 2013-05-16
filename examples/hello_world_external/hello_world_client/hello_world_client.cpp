//  Copyright (c) 2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//[hello_world_external_client_getting_started
#include <hpx/hpx_main.hpp>

#include "hello_world_component/hello_world_component.hpp"

int main()
{
    // Create a single instance of the component on this locality.
    examples::hello_world client;
    client.create(hpx::find_here());

    // Invoke the components action, which will print "Hello World!".
    client.invoke();

    return 0;
}
//]

