//  Copyright (c) 2007-2011 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach 
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/lcos/barrier.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/atomic.hpp>

// TODO: Modernize this and make it test barrier in distributed.

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using hpx::naming::id_type;

using hpx::applier::get_applier;
using hpx::applier::register_work;

using hpx::lcos::barrier;

using hpx::init;
using hpx::finalize;

using hpx::util::report_errors;

///////////////////////////////////////////////////////////////////////////////
void barrier_test(id_type const& id, boost::atomic<std::size_t>& c)
{
    ++c;
    // wait for all threads to enter the barrier
    hpx::lcos::stubs::barrier::wait(id);
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm)
{
//    std::size_t num_threads = 1;

//    if (vm.count("threads"))
//        num_threads = vm["threads"].as<std::size_t>();

    std::size_t pxthreads = 0;

    if (vm.count("pxthreads"))
        pxthreads = vm["pxthreads"].as<std::size_t>();
    
    std::size_t iterations = 0;

    if (vm.count("iterations"))
        iterations = vm["iterations"].as<std::size_t>();

    for (std::size_t i = 0; i < iterations; ++i)
    {
        id_type prefix = get_applier().get_runtime_support_gid();

        // create a barrier waiting on 'count' threads
        barrier b;
        b.create(prefix, pxthreads + 1);

        boost::atomic<std::size_t> c(0);

        for (std::size_t j = 0; j < pxthreads; ++j)
            register_work(boost::bind 
                (&barrier_test, b.get_gid(), boost::ref(c)));

        b.wait(); // wait for all threads to enter the barrier
        HPX_TEST_EQ(pxthreads, c.load());
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

