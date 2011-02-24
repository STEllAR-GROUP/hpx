//  Copyright (c) 2007-2011 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach 
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/util/lightweight_test.hpp>
#include <hpx/lcos/barrier.hpp>

#include <boost/preprocessor/stringize.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/detail/atomic_count.hpp>
#include <boost/program_options.hpp>

using namespace hpx;
namespace po = boost::program_options;

///////////////////////////////////////////////////////////////////////////////
void barrier_test(naming::id_type const& id, boost::detail::atomic_count& c, std::size_t count)
{
    ++c;
    lcos::stubs::barrier::wait(id);// wait for all threads to enter the barrier
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(po::variables_map &vm)
{
    int num_threads = 1, num_tests = 1;

    if (vm.count("threads"))
        num_threads = vm["threads"].as<int>();

    if (vm.count("num_tests"))
        num_tests = vm["num_tests"].as<int>();
    
    naming::id_type prefix = applier::get_applier().get_runtime_support_gid();

    for (int t = 0; t < num_tests; ++t) {
        std::size_t count = 2 * num_threads;

        // create a barrier waiting on 'count' threads
        lcos::barrier b;
        b.create_one (prefix, count+1);

        boost::detail::atomic_count c(0);
        for (std::size_t j = 0; j < count; ++j) {
            applier::register_work(
                boost::bind(&barrier_test, b.get_gid(), boost::ref(c), count));
        }

        b.wait(); // wait for all threads to enter the barrier
        HPX_TEST_EQ(count, c);
    }

    // initiate shutdown of the runtime system
    hpx::finalize();
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    po::options_description
       desc_commandline
          ("usage: " BOOST_PP_STRINGIZE(HPX_APPLICATION_NAME) " [options]");
        
    desc_commandline.add_options()
        ("num_tests,n", po::value<int>(), 
            "the number of times to repeat the test (default: 1)")
        ;

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(desc_commandline, argc, argv), 0,
      "HPX main exited with non-zero status");
    return boost::report_errors();
}

