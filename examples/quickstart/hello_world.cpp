///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#include <hpx/hpx_init.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/iostreams.hpp>

#include <vector>
#include <list>
#include <set>

#include <boost/ref.hpp>
#include <boost/foreach.hpp>
#include <boost/format.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;

using boost::format;

using hpx::init;
using hpx::finalize;
using hpx::get_os_thread_count;
using hpx::find_here;
using hpx::find_all_localities;

using hpx::naming::id_type;

using hpx::actions::plain_action0;
using hpx::actions::plain_result_action1;

using hpx::lcos::promise;
using hpx::lcos::async;
using hpx::lcos::wait;

using hpx::cout;
using hpx::flush;

///////////////////////////////////////////////////////////////////////////////
// The purpose of this example is to execute a PX-thread printing "Hello world"
// once on each OS-thread on each of the connected localities.
//
// The function hello_world_foreman_action is executed once on each locality.
// It schedules a PX-thread (encapsulating hello_world_worker) once for each
// OS-thread on that locality. The code make sure that the PX-thread gets
// really executed by the requested OS-thread. While the PX-thread is scheduled
// to run on a particular OS-thread, we may have to retry as the PX-thread may
// end up being 'stolen' by another OS-thread.

///////////////////////////////////////////////////////////////////////////////
std::size_t hello_world_worker(std::size_t desired)
{
    std::size_t current = hpx::get_worker_thread_num();

    if (current == desired)
    {
        // Yes! The PX-thread is run by the designated OS-thread.
        cout << ( format("hello world from OS-thread %1% on locality %2%\n")
                  % desired
                  % hpx::get_locality_id())
            << flush;
        return desired;
    }

    // this PX-thread is run by the wrong OS-thread, make the foreman retry
    return std::size_t(-1);
}

// Define the boilerplate code necessary for the function 'hello_world_worker'
// to be invoked as an HPX action (by a HPX future)
typedef plain_result_action1<std::size_t, std::size_t, hello_world_worker>
    hello_world_worker_action;

HPX_REGISTER_PLAIN_ACTION(hello_world_worker_action);

///////////////////////////////////////////////////////////////////////////////
void hello_world_foreman()
{
    std::size_t const os_threads = get_os_thread_count();
    id_type const prefix = find_here();

    std::set<std::size_t> attendance;
    for (std::size_t os_thread = 0; os_thread < os_threads; ++os_thread)
        attendance.insert(os_thread);

    // Retry until all PX-threads got executed by their designated OS-thread.
    while (!attendance.empty())
    {
        std::vector<promise<std::size_t> > futures;
        futures.reserve(attendance.size());
        BOOST_FOREACH(std::size_t os_thread, attendance)
        {
            // Schedule a PX-thread encapsulating the print action on a
            // particular OS-thread.
            futures.push_back(async<hello_world_worker_action>(prefix, os_thread));
        }

        // wait for all of the futures to return their values, we re-spawn the
        // future until the action gets executed on the right OS-thread
        wait(futures, [&](std::size_t, std::size_t t)
            { if (std::size_t(-1) != t) attendance.erase(t); });
    }
}

// Define the boilerplate code necessary for the function 'hello_world_foreman'
// to be invoked as an HPX action
//[hello_world_action_wrapper
typedef plain_action0<hello_world_foreman> hello_world_foreman_action;

HPX_REGISTER_PLAIN_ACTION(hello_world_foreman_action);
//]

///////////////////////////////////////////////////////////////////////////////
//[hello_world_hpx_main
//`Here is hpx_main:
int hpx_main(variables_map&)
{
    {
        std::vector<id_type> prefixes = find_all_localities();/*<Returns the number
                                                                 of localities>*/

        std::vector<promise<void> > futures;/*<A promise is a chunk of work which
                                               is executed after being assigned>*/
        futures.reserve(prefixes.size());
        BOOST_FOREACH(id_type const& node, prefixes)
        {
            futures.push_back(async<hello_world_foreman_action>(node)); /*<
            asyncronoulsy call hello_world_foreman_action>*/
        }

        wait(futures);    /* Wait for all IO to finish */ /*<
        Wait() will cause the OS thread to wait for all the promises stored
           in futures to be executed before continuing the program>*/
    }

    // Initiate shutdown of the runtime system.
    finalize();

    return 0;
}
//]

///////////////////////////////////////////////////////////////////////////////
//[hello_world_main
int main(int argc, char* argv[])
{
    // Configure application-specific options.
    options_description
       desc_commandline("usage: " HPX_APPLICATION_STRING " [options]");

    // Initialize and run HPX.
    return init(desc_commandline, argc, argv);
}
/*`
  In HPX main is used to initialize the runtime system and pass the command line
  arguments to the program.  If you wish to add command line options to your
  program you would add them here using desc_commandline.add_options() [fixme link
  to API] (see the API for more details).  Init() calls hpx_main after setting up
  HPX, which is where the mechanics of our program is written.
 */
//]
