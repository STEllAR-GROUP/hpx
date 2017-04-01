////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2017 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <chrono>
#include <cstddef>
#include <thread>

void f(int, float, double) {}

int main()
{
    // check thread
    {
        std::thread t;
        t = std::thread(&f, 1, 2.0f, 3.);

        if (t.joinable())
            t.join();

        std::size_t hwc = std::thread::hardware_concurrency();
    }

    // check thread::id
    {
        std::thread::id const id;
        id < id; // as map keys
    }

    // check this_thread namespace
    {
        std::thread::id const id = std::this_thread::get_id();

        std::this_thread::yield();

        std::this_thread::sleep_for(std::chrono::seconds(0));
        std::this_thread::sleep_until(std::chrono::system_clock::now());
    }
}
