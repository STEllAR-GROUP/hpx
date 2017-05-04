////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2017 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <thread>

void f(int, float, double) {}

bool pred() { return true; }

int main()
{
    std::chrono::seconds rel_time;
    std::chrono::system_clock::time_point abs_time;

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

        std::this_thread::sleep_for(rel_time);
        std::this_thread::sleep_until(abs_time);
    }

    // check mutex
    {
        std::mutex mut;
        mut.lock();
        bool c = mut.try_lock();
        mut.unlock();
    }

    // check recursive_mutex
    {
        std::recursive_mutex mut;
        mut.lock();
        bool c = mut.try_lock();
        mut.unlock();
    }

    // check condition_variable
    {
        std::condition_variable cv;

        cv.notify_one();
        cv.notify_all();

        std::mutex mut;
        std::unique_lock<std::mutex> lock(mut);
        cv.wait(lock);
        cv.wait(lock, pred);
        cv.wait_for(lock, rel_time);
        cv.wait_for(lock, rel_time, pred);
        cv.wait_until(lock, abs_time);
        cv.wait_until(lock, abs_time, pred);
    }
}
