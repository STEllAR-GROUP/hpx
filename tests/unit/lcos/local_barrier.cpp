//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/include/local_lcos.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/atomic.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using hpx::applier::register_work;

using hpx::lcos::local::barrier;

using hpx::init;
using hpx::finalize;

using hpx::util::report_errors;

///////////////////////////////////////////////////////////////////////////////
void local_barrier_test(barrier& b, boost::atomic<std::size_t>& c)
{
    ++c;
    // wait for all threads to enter the barrier
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

    for (std::size_t i = 0; i < iterations; ++i)
    {
        // create a barrier waiting on 'count' threads
        barrier b(pxthreads + 1);

        boost::atomic<std::size_t> c(0);

        // create the threads which will wait on the barrier
        for (std::size_t i = 0; i < pxthreads; ++i)
            register_work(boost::bind
                (&local_barrier_test, boost::ref(b), boost::ref(c)));

        b.wait(); // wait for all threads to enter the barrier
        HPX_TEST_EQ(pxthreads, c);
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
        ("pxthreads,T", value<std::size_t>()->default_value(64),
            "the number of PX threads to invoke")
        ("iterations", value<std::size_t>()->default_value(64),
            "the number of times to repeat the test")
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

