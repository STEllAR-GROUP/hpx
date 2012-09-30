//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/util/lightweight_test.hpp>
#include <hpx/lcos/local/event_semaphore.hpp>

#include <boost/atomic.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using hpx::applier::register_work;

using hpx::lcos::local::event_semaphore;

using hpx::init;
using hpx::finalize;

using hpx::util::report_errors;

///////////////////////////////////////////////////////////////////////////////
void local_event_test(event_semaphore& b, boost::atomic<std::size_t>& c)
{
    ++c;
    // Wait for the event to occur.
    b.wait();
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm)
{
    std::size_t num_threads = 1;

    if (vm.count("threads"))
        num_threads = vm["threads"].as<std::size_t>();

    std::size_t pxthreads = 0;

    if (vm.count("pxthreads"))
        pxthreads = vm["pxthreads"].as<std::size_t>();

    std::size_t iterations = 0;

    if (vm.count("iterations"))
        iterations = vm["iterations"].as<std::size_t>();

    for (std::size_t i = 0; i < iterations; ++i)
    {
        event_semaphore e; 

        boost::atomic<std::size_t> c(0);

        // Create the threads which will wait on the event
        for (std::size_t i = 0; i < pxthreads; ++i)
            register_work(boost::bind
                (&local_event_test, boost::ref(e), boost::ref(c)));

        // Release all the threads.
        e.set(); 
        HPX_TEST_EQ(pxthreads, c);

        // Make sure that waiting on a set event works.
        e.wait();
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
        ("pxthreads,T", value<std::size_t>()->default_value(64),
            "the number of PX threads to invoke")
        ("iterations", value<std::size_t>()->default_value(64),
            "the number of times to repeat the test")
        ;

    // We force this test to use several threads by default.
    using namespace boost::assign;
    std::vector<std::string> cfg;
    cfg += "hpx.os_threads=" +
        boost::lexical_cast<std::string>(hpx::hardware_concurrency());

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(init(desc_commandline, argc, argv, cfg), 0,
      "HPX main exited with non-zero status");
    return report_errors();
}

