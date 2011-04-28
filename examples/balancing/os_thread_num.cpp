//  Copyright (c) 2011 Bryce Lelbach 
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>

#include <boost/lockfree/fifo.hpp>

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/lcos/local_barrier.hpp>

using boost::lockfree::fifo;

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using hpx::lcos::local_barrier;

using hpx::threads::threadmanager_base;

using hpx::applier::register_work;

using hpx::init;
using hpx::finalize;

///////////////////////////////////////////////////////////////////////////////
void get_os_thread_num(local_barrier& barr, fifo<std::size_t>& shepherds)
{
    shepherds.enqueue(threadmanager_base::get_thread_num());
    barr.wait();    
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm)
{
    std::size_t pxthreads = 0;

    if (vm.count("pxthreads"))
        pxthreads = vm["pxthreads"].as<std::size_t>();
    
    std::size_t iterations = 0;

    if (vm.count("iterations"))
        iterations = vm["iterations"].as<std::size_t>();

    for (std::size_t i = 0; i < iterations; ++i)
    {
        std::cout << "iteration: " << i << "\n";

        // Have the fifo preallocate the nodes.
        fifo<std::size_t> shepherds(pxthreads); 

        local_barrier barr(pxthreads + 1);

        for (std::size_t j = 0; j < pxthreads; ++j)
            register_work(boost::bind 
                (&get_os_thread_num, boost::ref(barr), boost::ref(shepherds)));

        barr.wait(); // wait for all PX threads to enter the barrier

        std::size_t shepherd = 0;

        while (shepherds.dequeue(&shepherd))
            std::cout << "  " << shepherd << "\n"; 
    }

    // initiate shutdown of the runtime system
    finalize();
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    options_description
       desc_commandline("usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ("pxthreads,T", value<std::size_t>()->default_value(128), 
            "the number of PX threads to invoke")
        ("iterations", value<std::size_t>()->default_value(1), 
            "the number of times to repeat the test") 
        ;

    // Initialize and run HPX
    return init(desc_commandline, argc, argv);
}

