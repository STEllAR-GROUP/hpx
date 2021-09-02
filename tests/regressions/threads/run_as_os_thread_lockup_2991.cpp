//  Copyright (c) 2017 Maciej Brodowicz
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/local/init.hpp>
#include <hpx/runtime_local/run_as_os_thread.hpp>

#include <iostream>
#include <mutex>
#include <thread>

std::mutex mtx;

void locker()
{
    std::cout << std::this_thread::get_id() << ": about to lock mutex\n";
    std::lock_guard<std::mutex> lock(mtx);
    std::cout << std::this_thread::get_id() << ": mutex locked\n";
}

int hpx_main()
{
    {
        std::cout << std::this_thread::get_id() << ": about to lock mutex\n";
        std::lock_guard<std::mutex> lock(mtx);
        std::cout << std::this_thread::get_id() << ": mutex locked\n";

        std::cout << std::this_thread::get_id()
                  << ": about to run on io thread\n";
        hpx::threads::run_as_os_thread(locker);
        //sleep(2);
    }
    std::cout << std::this_thread::get_id() << ": exiting\n";

    return hpx::local::finalize();
}

int main(int argc, char** argv)
{
    return hpx::local::init(hpx_main, argc, argv);
}
