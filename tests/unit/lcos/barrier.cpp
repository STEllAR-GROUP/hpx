//  Copyright (c) 2007-2011 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach 
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/thread/thread.hpp>

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/lcos/barrier.hpp>
#include <hpx/util/lightweight_test.hpp>

///////////////////////////////////////////////////////////////////////////////
void barrier_test(hpx::naming::id_type const& id,
                  boost::detail::atomic_count& c)
{
    std::cout << "Entered barrier on OS thread "
              << boost::this_thread::get_id() << std::endl;
    ++c;
    // wait for all threads to enter the barrier
    hpx::lcos::stubs::barrier::wait(id);
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map &vm)
{
    std::size_t num_threads = 1, value = num_threads * 2;

    if (vm.count("value"))
        value = vm["value"].as<std::size_t>();

    if (vm.count("threads"))
        num_threads = vm["threads"].as<std::size_t>();
    
    std::cout << "Number of OS threads: " << num_threads << std::endl;

    hpx::naming::id_type prefix =
        hpx::applier::get_applier().get_runtime_support_gid();

    // create a barrier waiting on 'count' threads
    hpx::lcos::barrier b;
    b.create_one(prefix, value);

    boost::detail::atomic_count c(0);

    for (std::size_t j = 0; j < value; ++j)
        hpx::applier::register_work
            (boost::bind(&barrier_test, b.get_gid(), boost::ref(c)));

    b.wait(); // wait for all threads to enter the barrier
    HPX_TEST_EQ(value, c);

    // initiate shutdown of the runtime system
    hpx::finalize();
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    boost::program_options::options_description
       desc_commandline("usage: " HPX_APPLICATION_STRING " [options]");
        
    desc_commandline.add_options()
        ("value,v", boost::program_options::value<std::size_t>(), 
            "the number of threads to wait on (default: OS threads * 2)")
        ;

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(desc_commandline, argc, argv), 0,
      "HPX main exited with non-zero status");
    return hpx::util::report_errors();
}

