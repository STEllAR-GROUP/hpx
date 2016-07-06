//  Copyright (c) 2016 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "../../../hpx/concurrent/concurrent_vector.hpp"

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <iostream>
#include <iomanip>
#include <functional>
#include <string>
#include <vector>

void unordered_vector_test1()
{
    // Create an unordered_map of three strings (that map to strings)
    hpx::concurrent::concurrent_vector<int> u;

    for (int i=0; i<10000; ++i) {
        u.push_back(i);
    }
    HPX_TEST(u.size()==10000);
}
///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    unordered_vector_test1();

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // By default this test should run on all available cores
    std::vector<std::string> cfg;
    cfg.push_back("hpx.os_threads=" +
        std::to_string(hpx::threads::hardware_concurrency()));

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}


