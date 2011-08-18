//  Copyright (c) 2007-2010 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/util/lightweight_test.hpp>
#include <hpx/lcos/local_barrier.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using hpx::naming::id_type;

using hpx::applier::get_applier;
using hpx::applier::register_work;

using hpx::threads::get_self;
using hpx::threads::thread_state_ex;

using hpx::lcos::local_barrier;

using hpx::util::full_empty;
using hpx::util::report_errors;

using hpx::init;
using hpx::finalize;

///////////////////////////////////////////////////////////////////////////////
void full_empty_test_helper(local_barrier& barr, full_empty<int>& data)
{
    // retrieve gid for this thread
    id_type gid = get_applier().get_thread_manager().
        get_thread_gid(get_self().get_thread_id());
    HPX_TEST(gid);

    data.set(1);
    HPX_TEST(!data.is_empty());

    barr.wait();
}

void full_empty_test(local_barrier& barr)
{
    // retrieve gid for this thread
    id_type gid = get_applier().get_thread_manager().
        get_thread_gid(get_self().get_thread_id());
    HPX_TEST(gid);

    // create a full_empty data item
    full_empty<int> data;
    HPX_TEST(data.is_empty());

    // schedule the helper thread
    register_work(boost::bind
        (&full_empty_test_helper, boost::ref(barr), boost::ref(data)));

    // wait for the other thread to set 'data' to full
    int value = 0;
    data.read(value); // this blocks for full_empty_test_helper to set the value

    HPX_TEST(!data.is_empty());
    HPX_TEST_EQ(value, 1);

    value = 0;
    data.read(value); // this should not block anymore

    HPX_TEST(!data.is_empty());
    HPX_TEST_EQ(value, 1);

    barr.wait();
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm)
{
    std::size_t iterations = 0;

    if (vm.count("iterations"))
        iterations = vm["iterations"].as<std::size_t>();

    local_barrier barr((iterations * 2) + 1);

    for (std::size_t i = 0; i < iterations; ++i)
        // schedule test threads
        register_work(boost::bind(&full_empty_test, boost::ref(barr)));

    barr.wait();

    // initiate shutdown of the runtime system
    finalize();
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options.
    options_description
        desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");
    
    desc_commandline.add_options()
        ("iterations", value<std::size_t>()->default_value(1 << 6), 
            "the number of times to repeat the test") 
        ;

    // Initialize and run HPX.
    HPX_TEST_EQ_MSG(init(desc_commandline, argc, argv), 0,
      "HPX main exited with non-zero status");
    return report_errors();
}

