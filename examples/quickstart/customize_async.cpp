//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// The purpose of this example is to demonstrate how to customize certain
// parameters (such like thread priority, the stacksize, or the targeted
// processing unit) for a thread which is created by calling hpx::apply() or
// hpx::async().

#include <hpx/hpx.hpp>
#include <hpx/hpx_main.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/include/thread_executors.hpp>

#include <algorithm>

///////////////////////////////////////////////////////////////////////////////
void run_with_large_stack()
{
    int const array_size = 1000000;

    // Allocating a huge array on the stack would normally cause problems.
    // For this reason, this function is scheduled on a thread using a large
    // stack (see below).
    char large_array[array_size];      // allocate 1 MByte of memory

    std::fill(large_array, &large_array[array_size], '\0');

    hpx::cout << "This thread runs with a "
              << hpx::threads::get_stack_size_name(
                    hpx::this_thread::get_stack_size())
              << " stack and "
              << hpx::threads::get_thread_priority_name(
                    hpx::this_thread::get_priority())
              << " priority." << hpx::endl;
}

///////////////////////////////////////////////////////////////////////////////
void run_with_high_priority()
{
    hpx::cout << "This thread runs with "
              << hpx::threads::get_thread_priority_name(
                    hpx::this_thread::get_priority())
              << " priority." << hpx::endl;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // run a thread on a large stack
    {
        hpx::threads::executors::default_executor large_stack_executor(
            hpx::threads::thread_stacksize_large);

        hpx::future<void> f =
            hpx::async(large_stack_executor, &run_with_large_stack);
        f.wait();
    }

    // run a thread with high priority
    {
        hpx::threads::executors::default_executor high_priority_executor(
            hpx::threads::thread_priority_critical);

        hpx::future<void> f =
            hpx::async(high_priority_executor, &run_with_high_priority);
        f.wait();
    }

    // combine both
    {
        hpx::threads::executors::default_executor fancy_executor(
            hpx::threads::thread_priority_critical,
            hpx::threads::thread_stacksize_large);

        hpx::future<void> f =
            hpx::async(fancy_executor, &run_with_large_stack);
        f.wait();
    }

    return 0;
}

