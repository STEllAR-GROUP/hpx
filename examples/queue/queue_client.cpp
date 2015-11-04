//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include <hpx/include/lcos.hpp>

#include <iostream>

///////////////////////////////////////////////////////////////////////////////
typedef hpx::lcos::queue<int> queue_type;

typedef hpx::components::component<
    hpx::lcos::server::queue<int>
> queue_of_ints_type;

HPX_REGISTER_DERIVED_COMPONENT_FACTORY(
    queue_of_ints_type, queue_of_ints_type,
    "hpx::lcos::base_lco_with_value<int, int>");

///////////////////////////////////////////////////////////////////////////////
void worker(queue_type queue)
{
    try {
        // retrieve one value, will possibly throw
        int value = queue.get_value_sync();
        std::cout << value << std::endl;
    }
    catch (hpx::exception const& e) {
        std::cout << e.what() << std::endl;
    }
}

void break_queue(queue_type queue)
{
    queue.abort_pending();
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map &vm)
{
    // Create a new queue of integers.
    queue_type queue = hpx::new_<queue_type>(hpx::find_here());

    // Create some threads waiting to pull elements from the queue.
    for (int i = 0; i < 5; ++i)
        hpx::apply(hpx::util::bind(&worker, queue));

    // Add some values to the queue.
    for (int i = 0; i < 5; ++i)
        queue.set_value_sync(i);

    // Create some threads waiting to pull elements from the queue, these
    // requests will fail because of the abort_pending() invoked below.
    for (int i = 0; i < 5; ++i)
        hpx::apply(hpx::util::bind(&worker, queue));

    hpx::apply(hpx::util::bind(&break_queue, queue));

    hpx::finalize();
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    return hpx::init("queue_of_ints_client", argc, argv);
}
