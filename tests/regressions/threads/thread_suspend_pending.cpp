//  Copyright (c) 2007-2011 Hartmut Kaiser
//  Copyright (c) 2011-2012 Bryce Adelstein-Lelbach 
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/util/lightweight_test.hpp>
#include <hpx/lcos/local/barrier.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using hpx::applier::register_work;

using hpx::lcos::local::barrier;

using hpx::init;
using hpx::finalize;

using hpx::threads::pending;

using hpx::util::report_errors;

///////////////////////////////////////////////////////////////////////////////
void suspend_test(barrier& b, std::size_t iterations)
{
    for (std::size_t i = 0; i < iterations; ++i)
    {
        // Enter the 'pending' state and get rescheduled. 
        hpx::this_thread::suspend(pending, "suspend_test");
    }

    // Wait for all hpx-threads to enter the barrier.
    b.wait();
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

    {
        barrier b(pxthreads + 1);

        // Create the hpx-threads.
        for (std::size_t i = 0; i < pxthreads; ++i)
            register_work(boost::bind
                (&suspend_test, boost::ref(b), iterations));

        b.wait(); // Wait for all hpx-threads to enter the barrier.
    }

    // Initiate shutdown of the runtime system.
    return finalize();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    options_description
       desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ("pxthreads,T", value<std::size_t>()->default_value(0x1000), 
            "the number of PX threads to invoke")
        ("iterations", value<std::size_t>()->default_value(64), 
            "the number of iterations to execute in each thread")
        ;

    // We force this test to use several threads by default.
    using namespace boost::assign;
    std::vector<std::string> cfg;
    cfg += "hpx.os_threads=" +
        boost::lexical_cast<std::string>(hpx::threads::hardware_concurrency());

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(init(desc_commandline, argc, argv, cfg), 0,
      "HPX main exited with non-zero status");
    return report_errors();
}

