//  Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
//  Copyright (c) 2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/apply.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/atomic.hpp>

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
boost::atomic<int> count(0);

void worker(hpx::lcos::local::counting_semaphore& sem)
{
    sem.signal();   // signal main thread
    ++count;
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    hpx::lcos::local::counting_semaphore sem;

    for (std::size_t i = 0; i != 10; ++i)
        hpx::apply(&worker, std::ref(sem));

    // Wait for all threads to finish executing.
    sem.wait(10);

    HPX_TEST(count == 10);

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // By default this test should run on all available cores
    std::vector<std::string> const cfg = {
        "hpx.os_threads=all"
    };

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
