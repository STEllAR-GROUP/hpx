//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>
#include <boost/bind.hpp>

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include <hpx/lcos/queue.hpp>

typedef hpx::lcos::queue<int> queue_type;

///////////////////////////////////////////////////////////////////////////////
void worker(queue_type queue)
{
    try {
        int value = queue.get_value_sync();   // retrieve one value, will possibly throw
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
    // create a new queue of integers
    queue_type queue;
    queue.create(hpx::find_here());

    // create some threads waiting to pull elements from the queue
    for (int i = 0; i < 5; ++i)
        hpx::applier::register_work(boost::bind(&worker, queue));

    // add some values to the queue
    for (int i = 0; i < 5; ++i)
        queue.set_value_sync(i);

    // create some threads waiting to pull elements from the queue
    for (int i = 0; i < 5; ++i)
        hpx::applier::register_work(boost::bind(&worker, queue));

    hpx::applier::register_work(boost::bind(&break_queue, queue));

    hpx::finalize();
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    return hpx::init("queue_of_ints_client", argc, argv); // Initialize and run HPX
}
