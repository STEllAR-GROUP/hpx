//  Copyright (c) 2011 Bryce Lelbach 
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>

#include <boost/lockfree/fifo.hpp>

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/lcos/local_barrier.hpp>

using boost::lockfree::fifo;

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using hpx::lcos::local_barrier;

using hpx::applier::register_thread;

using hpx::threads::get_thread_phase;
using hpx::threads::thread_id_type;

using hpx::init;
using hpx::finalize;

///////////////////////////////////////////////////////////////////////////////
void yield_once(local_barrier& _0)
{
    _0.wait(); // yield once, wait for hpx_main
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm)
{
    std::size_t pxthread_count = 0;

    if (vm.count("pxthreads"))
        pxthread_count = vm["pxthreads"].as<std::size_t>();
    
    std::size_t iterations = 0;

    if (vm.count("iterations"))
        iterations = vm["iterations"].as<std::size_t>();

    for (std::size_t i = 0; i < iterations; ++i)
    {
        std::cout << "iteration: " << i << "\n";

        // Have the fifo preallocate the nodes.
        fifo<thread_id_type> pxthreads(pxthread_count); 

        local_barrier barr(pxthread_count + 1);

        for (std::size_t j = 0; j < pxthread_count; ++j)
            pxthreads.enqueue(register_thread(boost::bind 
                (&yield_once, boost::ref(barr)), "yield_once"));

        barr.wait(); // wait for all PX threads to enter the barrier

        thread_id_type pxthread = 0;

        while (pxthreads.dequeue(&pxthread))
            std::cout << pxthread << " -> "
                      << get_thread_phase(pxthread) << "\n"; 
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
       desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ("pxthreads,T", value<std::size_t>()->default_value(128), 
            "the number of PX threads to invoke")
        ("iterations", value<std::size_t>()->default_value(1), 
            "the number of times to repeat the test") 
        ;

    // Initialize and run HPX
    return init(desc_commandline, argc, argv);
}

