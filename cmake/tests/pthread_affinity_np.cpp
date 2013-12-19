////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <pthread.h>

#include <boost/detail/lightweight_test.hpp>

// Based on the example in pthread_setaffinity(3).

int main()
{
    int s;
    cpu_set_t cpuset;
    pthread_t thread;

    thread = pthread_self();

    // Set affinity mask to include CPUs 0 to 7 (if available - otherwise, uses
    // 0 to N where n is the number of cores on the machine.
    CPU_ZERO(&cpuset);
    for (int j = 0; j < 8; ++j)
        CPU_SET(j, &cpuset);

    BOOST_TEST_EQ(pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset), 0);

    // Check the actual affinity mask assigned to the thread.
    BOOST_TEST_EQ(pthread_getaffinity_np(thread, sizeof(cpu_set_t), &cpuset), 0);

    std::cout << "set returned by pthread_getaffinity_np() contained:\n";
    for (int j = 0; j < CPU_SETSIZE; ++j)
        if (CPU_ISSET(j, &cpuset))
            std::cout << "CPU" << j << "\n";

    return boost::report_errors();
}

