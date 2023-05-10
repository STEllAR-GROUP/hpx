//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/barrier.hpp>
#include <hpx/functional.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/concurrency.hpp>
#include <hpx/mutex.hpp>
#include <hpx/thread.hpp>

#include <chrono>
#include <cstddef>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

using hpx::lockfree::queue;

using hpx::program_options::options_description;
using hpx::program_options::value;
using hpx::program_options::variables_map;

using std::chrono::milliseconds;

using hpx::barrier;
using hpx::mutex;

using hpx::threads::make_thread_function_nullary;
using hpx::threads::register_thread;
using hpx::threads::thread_init_data;

using hpx::threads::get_self;
using hpx::threads::get_self_id;
using hpx::threads::get_thread_phase;
using hpx::threads::set_thread_state;
using hpx::threads::thread_id_ref_type;
using hpx::threads::thread_id_type;

typedef std::pair<thread_id_type, std::size_t> value_type;
typedef std::vector<value_type> fifo_type;

///////////////////////////////////////////////////////////////////////////////
void lock_and_wait(mutex& m, std::shared_ptr<barrier<>> b0,
    std::shared_ptr<barrier<>> b1, value_type& entry, std::size_t /* wait */)
{
    // Wait for all hpxthreads in this iteration to be created.
    b0->arrive_and_wait();

    // keep this thread alive while being suspended
    thread_id_ref_type this_ = get_self_id();

    while (true)
    {
        // Try to acquire the mutex.
        std::unique_lock<mutex> l(m, std::try_to_lock);

        if (l.owns_lock())
        {
            entry = value_type(this_.noref(), get_thread_phase(this_.noref()));
            break;
        }

        // Schedule a wakeup.
        set_thread_state(this_.noref(), milliseconds(30),
            hpx::threads::thread_schedule_state::pending);

        // Suspend this HPX thread.
        hpx::this_thread::suspend(
            hpx::threads::thread_schedule_state::suspended);
    }

    // Make hpx_main wait for us to finish.
    b1->arrive_and_wait();
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm)
{
    std::size_t hpxthread_count = 0;

    if (vm.count("hpxthreads"))
        hpxthread_count = vm["hpxthreads"].as<std::size_t>();

    std::size_t mutex_count = 0;

    if (vm.count("mutexes"))
        mutex_count = vm["mutexes"].as<std::size_t>();

    std::size_t iterations = 0;

    if (vm.count("iterations"))
        iterations = vm["iterations"].as<std::size_t>();

    std::size_t wait = 0;

    if (vm.count("wait"))
        wait = vm["wait"].as<std::size_t>();

    for (std::size_t i = 0; i < iterations; ++i)
    {
        std::cout << "iteration: " << i << "\n";

        // Have the fifo preallocate storage.
        fifo_type hpxthreads(hpxthread_count);

        // Allocate the mutexes.
        std::vector<mutex> m(mutex_count);
        std::shared_ptr<barrier<>> b0 =
            std::make_shared<barrier<>>(hpxthread_count + 1);
        std::shared_ptr<barrier<>> b1 =
            std::make_shared<barrier<>>(hpxthread_count + 1);

        // keep created threads alive while they are suspended
        std::vector<thread_id_ref_type> ids;
        for (std::size_t j = 0; j < hpxthread_count; ++j)
        {
            // Compute the mutex to be used for this thread.
            const std::size_t index = j % mutex_count;

            thread_init_data data(
                make_thread_function_nullary(hpx::bind(&lock_and_wait,
                    std::ref(m[index]), b0, b1, std::ref(hpxthreads[j]), wait)),
                "lock_and_wait");
            ids.push_back(register_thread(data));
        }

        // Tell all hpxthreads that they can start running.
        b0->arrive_and_wait();

        // Wait for all hpxthreads to finish.
        b1->arrive_and_wait();

        // {{{ Print results for this iteration.
        for (value_type& entry : hpxthreads)
        {
            std::cout << "  " << entry.first << "," << entry.second << "\n";
        }
        // }}}
    }

    // Initiate shutdown of the runtime system.
    hpx::local::finalize();
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options.
    options_description desc_commandline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    // clang-format off
    desc_commandline.add_options()
        ("hpxthreads,T", value<std::size_t>()->default_value(128),
            "the number of PX threads to invoke")
        ("mutexes,M", value<std::size_t>()->default_value(1),
            "the number of mutexes to use")
        ("wait", value<std::size_t>()->default_value(30),
            "the number of milliseconds to wait between each lock attempt")
        ("iterations", value<std::size_t>()->default_value(1),
            "the number of times to repeat the test")
        ;
    // clang-format on

    // Initialize and run HPX.
    hpx::local::init_params init_args;
    init_args.desc_cmdline = desc_commandline;

    return hpx::local::init(hpx_main, argc, argv, init_args);
}
