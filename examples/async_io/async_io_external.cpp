//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// The purpose of this example is to demonstrate how it is possible to
// schedule an IO task onto an external OS-thread in HPX and how to
// synchronize the result of this IO task with a waiting HPX thread.

#include <hpx/hpx_init.hpp>
#include <hpx/include/runtime.hpp>
#include <hpx/include/iostreams.hpp>

#include <iostream>
#include <memory>

///////////////////////////////////////////////////////////////////////////////
struct registration_wrapper
{
    registration_wrapper(hpx::runtime* rt, char const* name)
      : rt_(rt)
    {
        // Register this thread with HPX, this should be done once for
        // each external OS-thread intended to invoke HPX functionality.
        // Calling this function more than once will silently fail (will
        // return false).
        hpx::register_thread(rt_, name);
    }
    ~registration_wrapper()
    {
        // Unregister the thread from HPX, this should be done once in the
        // end before the external thread exists.
        hpx::unregister_thread(rt_);
    }

    hpx::runtime* rt_;
};

// this function will be executed by an HPX thread
void set_value(
    std::shared_ptr<hpx::lcos::local::promise<int> > p,
    int result)
{
    // notify the waiting HPX thread and return a value
    p->set_value(result);
}

// this function will be executed by a dedicated OS thread
void do_async_io(char const* string_to_write,
    std::shared_ptr<hpx::lcos::local::promise<int> > p,
    hpx::runtime* rt)
{
    // register this thread in order to be able to call HPX functionality
    registration_wrapper wrap(rt, "external-io");

    // This IO operation will possibly block the IO thread in the
    // kernel.
    std::cout << "OS-thread: " << string_to_write << std::endl;

    // Create an HPX thread to guarantee that the promise::set_value
    // function can be invoked safely.
    hpx::threads::register_thread(hpx::util::bind(&set_value, p, 0));
}

// This function will be executed by an HPX thread
int io(char const* string_to_write)
{
    std::shared_ptr<hpx::lcos::local::promise<int> > p =
        std::make_shared<hpx::lcos::local::promise<int> >();

    // Create a new external OS-thread and schedule the handler to
    // run on one of its OS-threads.
    boost::thread external_os_thread(
        hpx::util::bind(&do_async_io, string_to_write, p, hpx::get_runtime_ptr()));

    int result = p->get_future().get();

    // wait for the external thread to exit
    external_os_thread.join();

    return result;
}

int hpx_main()
{
    {
        // Initiate an asynchronous IO operation wait for it to complete without
        // blocking any of the HPX thread-manager threads.
        //
        // This will suspend the current HPX thread until the IO operation is
        // finished.
        int result = io("Write this string to std::cout");

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

