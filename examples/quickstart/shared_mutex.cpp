//  Copyright (c) 2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This example was released to the public domain by Stephan T. Lavavej
// (see: https://channel9.msdn.com/Shows/C9-GoingNative/GoingNative-40-Updated-STL-in-VS-2015-feat-STL)

#include <hpx/hpx_main.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/include/iostreams.hpp>

#include <boost/atomic.hpp>
#include <boost/chrono.hpp>
#include <boost/random.hpp>
#include <boost/thread/locks.hpp>

#include <ctime>
#include <mutex>
#include <vector>

int const writers = 3;
int const readers = 3;
int const cycles = 10;

using boost::chrono::milliseconds;

int main()
{
    std::vector<hpx::thread> threads;
    boost::atomic<bool> ready(false);
    hpx::lcos::local::shared_mutex stm;

    for (int i = 0; i < writers; ++i)
    {
        threads.emplace_back(
            [&ready, &stm, i]
            {
                boost::random::mt19937 urng(
                    static_cast<boost::uint32_t>(std::time(nullptr)));
                boost::random::uniform_int_distribution<int> dist(1, 1000);

                while (!ready) { /*** wait... ***/ }

                for (int j = 0; j < cycles; ++j)
                {
                    std::unique_lock<hpx::lcos::local::shared_mutex> ul(stm);

                    hpx::cout << "^^^ Writer " << i << " starting..." << std::endl;
                    hpx::this_thread::sleep_for(milliseconds(dist(urng)));
                    hpx::cout << "vvv Writer " << i << " finished." << std::endl;

                    ul.unlock();

                    hpx::this_thread::sleep_for(milliseconds(dist(urng)));
                }
            });
    }

    for (int i = 0; i < readers; ++i)
    {
        int k = writers + i;
        threads.emplace_back(
            [&ready, &stm, k, i]
            {
                boost::random::mt19937 urng(
                    static_cast<boost::uint32_t>(std::time(nullptr)));
                boost::random::uniform_int_distribution<int> dist(1, 1000);

                while (!ready) { /*** wait... ***/ }

                for (int j = 0; j < cycles; ++j)
                {
                    boost::shared_lock<hpx::lcos::local::shared_mutex> sl(stm);

                    hpx::cout << "Reader " << i << " starting..." << std::endl;
                    hpx::this_thread::sleep_for(milliseconds(dist(urng)));
                    hpx::cout << "Reader " << i << " finished." << std::endl;

                    sl.unlock();

                    hpx::this_thread::sleep_for(milliseconds(dist(urng)));
                }
            });
    }

    ready = true;
    for (auto& t: threads)
        t.join();

    return 0;
}

