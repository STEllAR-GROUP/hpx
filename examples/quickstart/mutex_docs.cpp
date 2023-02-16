//  Copyright (c) 2023 Dimitra Karatza
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This example is meant for inclusion in the documentation.

#include <hpx/future.hpp>
#include <hpx/hpx_main.hpp>
#include <hpx/mutex.hpp>
#include <iostream>

int main()
{
    hpx::mutex m;

    hpx::future<void> f1 = hpx::async([&m]() {
        m.lock();
        std::cout << "Thread 1 acquired the mutex" << std::endl;
        m.unlock();
    });

    hpx::future<void> f2 = hpx::async([&m]() {
        m.lock();
        std::cout << "Thread 2 acquired the mutex" << std::endl;
        m.unlock();
    });

    hpx::wait_all(f1, f2);

    return 0;
}
