//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// The purpose of this example is to demonstrate how it is possible to
// generically schedule an asynchronous IO task onto one of the IO-threads
// in HPX (which are OS-threads), how to synchronize the result of this IO
// task with a waiting HPX thread, and how to wrap all of this into a HPX
// action.

#include <hpx/hpx_init.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/include/runtime.hpp>
#include <hpx/include/thread_executors.hpp>

#include <string>

namespace detail
{
    // this function will be executed by a dedicated OS thread
    void do_async_io(std::string const& string_to_write,
        boost::shared_ptr<hpx::lcos::local::promise<int> > p)
    {
        // This IO operation will possibly block the IO thread in the
        // kernel.
        std::cout << "OS-thread: " << string_to_write << std::endl;

        p->set_value(0);    // notify the waiting HPX thread and return a value
    }

    // This function will be executed by an HPX thread
    hpx::lcos::future<int> async_io_worker(std::string const& string_to_write)
    {
        boost::shared_ptr<hpx::lcos::local::promise<int> > p =
            boost::make_shared<hpx::lcos::local::promise<int> >();

        // Get a reference to one of the IO specific HPX io_service objects ...
        hpx::threads::executors::io_pool_executor scheduler;

        // ... and schedule the handler to run on one of its OS-threads.
        scheduler.add(hpx::util::bind(&do_async_io, string_to_write, p));

        return p->get_future();
    }
}

// This function will be called whenever the action async_io_action is
// invoked. This allows to remotely execute the async_io.
int async_io(std::string const& string_to_write)
{
    hpx::lcos::future<int> f = detail::async_io_worker(string_to_write);
    return f.get();    // simply wait for the IO to finish
}

// this defines the type async_io_action
HPX_PLAIN_ACTION(async_io, async_io_action);

int hpx_main()
{
    {
        // Determine on what locality to run the IO operation. If this
        // example is run on one locality, we perform the IO operation
        // locally, otherwise we choose one of the remote localities to
        // invoke the async_io_action.
        hpx::naming::id_type id = hpx::find_here();    // default: local

        std::vector<hpx::naming::id_type> localities =
            hpx::find_remote_localities();
        if (!localities.empty())
            id = localities[0];         // choose the first remote locality

        // Create an action instance.
        async_io_action io;

        // Initiate an asynchronous (possibly remote) IO operation and
        // wait for it to complete without blocking any of the HPX
        // thread-manager threads. This will suspend the current HPX
        // thread until the IO operation is finished. Other work could
        // be executed in the mean time.
        int result = io(id, "Write this string to std::cout");

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

