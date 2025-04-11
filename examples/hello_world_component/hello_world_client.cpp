//  Copyright (c) 2012 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Please keep the duplicate hpx/config.hpp include further down. It makes the
// code within the [hello_world_client... block self-contained for the
// documentation.
#include <hpx/config.hpp>
#if defined(HPX_WITH_DISTRIBUTED_RUNTIME)

//[hello_world_client_getting_started
#include <hpx/config.hpp>
#if defined(HPX_COMPUTE_HOST_CODE)
#include <hpx/wrap_main.hpp>

#include "hello_world_component.hpp"

int main()
{
    // Initialize HPX runtime
    hpx::init_params init_args;
    init_args.cfg = {"hpx.commandline.allow_unknown=1"};
    
    return hpx::init(init_args);
}
#endif
//]
#else
#include <hpx/thread.hpp>
#include <hpx/wrap_main.hpp>

#include <iostream>

int main()
{
    std::cout << "Hello World from HPX-thread with id "
              << hpx::this_thread::get_id() << std::endl;

    return 0;
}
#endif
