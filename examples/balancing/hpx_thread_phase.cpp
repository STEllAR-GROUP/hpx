//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/lcos/local/barrier.hpp>
#include <hpx/lcos/local/mutex.hpp>

#include <boost/chrono/duration.hpp>
#include <boost/lockfree/queue.hpp>

#include <iostream>

using boost::lockfree::queue;

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using boost::chrono::milliseconds;

using hpx::lcos::local::barrier;
using hpx::lcos::local::mutex;

using hpx::applier::register_thread;

using hpx::threads::get_thread_phase;
using hpx::threads::get_self_id;
using hpx::threads::get_self;
using hpx::threads::thread_id_type;
using hpx::threads::pending;
using hpx::threads::suspended;
using hpx::threads::set_thread_state;

using hpx::init;
using hpx::finalize;

typedef queue<std::pair<thread_id_type, std::size_t>*> fifo_type;

///////////////////////////////////////////////////////////////////////////////
void lock_and_wait(
    mutex& m
  , barrier& b0
  , barrier& b1
  , fifo_type& hpxthreads
  , std::size_t wait
) {
    // Wait for all hpxthreads in this iteration to be created.
    b0.wait();

    const thread_id_type this_ = get_self_id();

    while (true)
    {
        // Try to acquire the mutex.
        mutex::scoped_try_lock l(m);

        if (l.owns_lock())
        {
            hpxthreads.push(new std::pair<thread_id_type, std::size_t>
                (this_, get_thread_phase(this_)));
            break;
        }

        // Schedule a wakeup.
        set_thread_state(this_, milliseconds(30), pending);

        // Suspend this HPX thread.
        hpx::this_thread::suspend(suspended);
    }

    // Make hpx_main wait for us to finish.
    b1.wait();
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

        std::vector<mutex*> m(mutex_count, 0);
        barrier b0(hpxthread_count + 1), b1(hpxthread_count + 1);

        // Allocate the mutexes.
        for (std::size_t j = 0; j < mutex_count; ++j)
            m[j] = new mutex;

        for (std::size_t j = 0; j < hpxthread_count; ++j)
        {
            // Compute the mutex to be used for this thread.
            const std::size_t index = j % mutex_count;

            register_thread(boost::bind
                (&lock_and_wait, boost::ref(*m[index])
                               , boost::ref(b0)
                               , boost::ref(b1)
                               , boost::ref(hpxthreads)
                               , wait)
              , "lock_and_wait");
        }

        // Tell all hpxthreads that they can start running.
        b0.wait();

        // Wait for all hpxthreads to finish.
        b1.wait();

        // {{{ Print results for this iteration.
        std::pair<thread_id_type, std::size_t>* entry = 0;

        while (hpxthreads.pop(entry))
        {
            HPX_ASSERT(entry);
            std::cout << "  " << entry->first << "," << entry->second << "\n";
            delete entry;
        }
        // }}}

        // Destroy the mutexes.
        for (std::size_t j = 0; j < mutex_count; ++j)
        {
            HPX_ASSERT(m[j]);
            delete m[j];
        }
    }

    // Initiate shutdown of the runtime system.
    finalize();
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options.
    options_description
       desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

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

    // Initialize and run HPX.
    return init(desc_commandline, argc, argv);
}

