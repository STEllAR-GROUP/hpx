// Copyright (C) 2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/threadmanager.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <vector>
#include <string>

#include <boost/lexical_cast.hpp>

#define NUM_YIELD_TESTS 1000

///////////////////////////////////////////////////////////////////////////////
void test_yield()
{
    for (std::size_t i = 0; i != NUM_YIELD_TESTS; ++i)
        hpx::this_thread::yield();
}

int hpx_main()
{
    std::size_t num_cores = hpx::get_os_thread_count();

    std::vector<hpx::future<void> > finished;
    finished.reserve(num_cores);

    for (std::size_t i = 0; i != num_cores; ++i)
        finished.push_back(hpx::async(&test_yield));

    hpx::wait_all(finished);

    return hpx::finalize();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // By default this test should run on all available cores
    std::vector<std::string> cfg;
    cfg.push_back("hpx.os_threads=" +
        boost::lexical_cast<std::string>(hpx::threads::hardware_concurrency()));

    HPX_TEST_EQ(hpx::init(argc, argv, cfg), 0);
    return hpx::util::report_errors();
}

