//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#if defined(HPX_HAVE_QUEUE_COMPATIBILITY)
#include <hpx/include/lcos.hpp>

#include <iostream>

///////////////////////////////////////////////////////////////////////////////
typedef hpx::lcos::queue<int> queue_type;

HPX_REGISTER_QUEUE(int);

///////////////////////////////////////////////////////////////////////////////
void worker(queue_type queue)
{
    try {
        // retrieve one value, will possibly throw
        int value = queue.get_value(hpx::launch::sync);
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
#endif

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map &vm)
{
#if defined(HPX_HAVE_QUEUE_COMPATIBILITY)
    // Create a new queue of integers.
    queue_type queue = hpx::new_<queue_type>(hpx::find_here());

    // Create some threads waiting to pull elements from the queue.
    for (int i = 0; i < 5; ++i)
        hpx::apply(hpx::util::bind(&worker, queue));

    // Add some values to the queue.
    for (int i = 0; i < 5; ++i)
        queue.set_value(hpx::launch::sync, i);

    // Create some threads waiting to pull elements from the queue, these
    // requests will fail because of the abort_pending() invoked below.
    for (int i = 0; i < 5; ++i)
        hpx::apply(hpx::util::bind(&worker, queue));

    hpx::apply(hpx::util::bind(&break_queue, queue));
#endif

    return hpx::finalize();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    return hpx::init("queue_of_ints_client", argc, argv);
}

