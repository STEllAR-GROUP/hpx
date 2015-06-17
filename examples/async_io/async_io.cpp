//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// The purpose of this example is to demonstrate how it is possible to
// schedule an IO task onto one of the IO-threads in HPX (which are OS-threads)
// and how to synchronize the result of this IO task with a waiting HPX thread.

#include <hpx/hpx_init.hpp>
#include <hpx/include/runtime.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/include/thread_executors.hpp>

// this function will be executed by a dedicated OS thread
void do_async_io(char const* string_to_write, int& result)
{
    // This IO operation will possibly block the IO thread in the
    // kernel.
    std::cout << "OS-thread: " << string_to_write << std::endl;

    result = 0;
}

// This function will be executed by an HPX thread
int async_io(char const* string_to_write)
{
    int result = 0;

    {
        // Get a reference to one of the IO specific HPX io_service objects ...
        hpx::threads::executors::io_pool_executor scheduler;

        // ... and schedule the handler to run on one of its OS-threads.
        scheduler.add(hpx::util::bind(&do_async_io,
            string_to_write, boost::ref(result)));

    // Note that the destructor of the scheduler object will wait for
    // the scheduled task to finish executing. This might be
    // undesireable, in which case the technique demonstrated in
    // the example async_io_low_level.cpp is preferrable.
    }

    return result;   // this will be executed only after result has been set
}

int hpx_main()
{
    {
        // Execute an asynchronous IO operation wait for it to complete without
        // blocking any of the HPX thread-manager threads.
        int result = async_io("Write this string to std::cout");

        // Print the returned result.
        hpx::cout << "HPX-thread: The asynchronous IO operation returned: "
                  << result << "\n" << hpx::flush;
    }

    return hpx::finalize(); // Initiate shutdown of the runtime system.
}

int main(int argc, char* argv[])
{
    return hpx::init(argc, argv); // Initialize and run HPX.
}

