//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/threads.hpp>

#include <vector>

int const num_threads = 10;

///////////////////////////////////////////////////////////////////////////////
void wait_for_latch(hpx::lcos::local::latch& l)
{
    l.count_down_and_wait();
}

int main()
{
    // Spawn a couple of threads
    hpx::lcos::local::latch l(num_threads+1);

    std::vector<hpx::future<void> > results;
    results.reserve(num_threads);

    for (int i = 0; i != num_threads; ++i)
        results.push_back(hpx::async(&wait_for_latch, std::ref(l)));

    // Allow spawned threads to reach latch
    hpx::this_thread::yield();

    // Enumerate all suspended threads
    hpx::threads::enumerate_threads(
        [](hpx::threads::thread_id_type id) -> bool
        {
            hpx::cout
                << "thread "
                << hpx::thread::id(id) << " is "
                << hpx::threads::get_thread_state_name(
                        hpx::threads::get_thread_state(id))
                << std::endl;
            return true;        // always continue enumeration
        },
        hpx::threads::suspended);

    // Wait for all threads to reach this point.
    l.count_down_and_wait();

    hpx::wait_all(results);

    return 0;
}


