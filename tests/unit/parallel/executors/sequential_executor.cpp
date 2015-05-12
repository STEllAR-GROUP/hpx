//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_executors.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <algorithm>
#include <cstdlib>
#include <vector>

#include <boost/range/functions.hpp>

///////////////////////////////////////////////////////////////////////////////
hpx::thread::id test() { return hpx::this_thread::get_id(); }

void test_sync()
{
    typedef hpx::parallel::sequential_executor executor;
    typedef hpx::parallel::executor_traits<executor> traits;

    executor exec;
    HPX_TEST(traits::execute(exec, &test) == hpx::this_thread::get_id());
}

void test_async()
{
    typedef hpx::parallel::sequential_executor executor;
    typedef hpx::parallel::executor_traits<executor> traits;

    executor exec;
    HPX_TEST(
        traits::async_execute(exec, &test).get() ==
        hpx::this_thread::get_id());
}

///////////////////////////////////////////////////////////////////////////////
void bulk_test(hpx::thread::id tid, int value)
{
    HPX_TEST(tid == hpx::this_thread::get_id());
}

void test_bulk_sync()
{
    typedef hpx::parallel::sequential_executor executor;
    typedef hpx::parallel::executor_traits<executor> traits;

    hpx::thread::id tid = hpx::this_thread::get_id();

    std::vector<int> v(107);
    std::iota(boost::begin(v), boost::end(v), std::rand());

    using hpx::util::placeholders::_1;

    executor exec;
    traits::execute(exec, hpx::util::bind(&bulk_test, tid, _1), v);
}

int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int)std::time(0);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    test_sync();
    test_async();
    test_bulk_sync();

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // add command line option which controls the random number generator seed
    using namespace boost::program_options;
    options_description desc_commandline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ("seed,s", value<unsigned int>(),
        "the random number generator seed to use for this run")
        ;
    // By default this test should run on all available cores
    std::vector<std::string> cfg;
    cfg.push_back("hpx.os_threads=" +
        boost::lexical_cast<std::string>(hpx::threads::hardware_concurrency()));

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}

