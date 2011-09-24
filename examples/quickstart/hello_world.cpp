///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach 
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#include <hpx/hpx_init.hpp>
#include <hpx/lcos/eager_future.hpp>
#include <hpx/lcos/future_wait.hpp>
#include <hpx/lcos/async_future_wait.hpp>
#include <hpx/runtime/actions/continuation_impl.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/components/plain_component_factory.hpp>
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
using hpx::get_num_os_threads;
using hpx::find_here;
using hpx::find_all_localities;

using hpx::naming::id_type;

using hpx::applier::get_prefix_id;

using hpx::actions::plain_action0;
using hpx::actions::plain_result_action1;

using hpx::lcos::future_value;
using hpx::lcos::eager_future;
using hpx::lcos::wait;

using hpx::threads::threadmanager_base;

using hpx::cout;
using hpx::endl;

///////////////////////////////////////////////////////////////////////////////
std::size_t hello_world_worker(std::size_t desired)
{
    std::size_t current = threadmanager_base::get_thread_num();

    if (current == desired)
    {
        cout << ( format("hello world from shepherd %1% on locality %2%")
                  % desired 
                  % get_prefix_id())
            << endl;
        return desired;
    }

    return std::size_t(-1);
}

// Define the boilerplate code necessary for the function 'hello_world_worker'
// to be invoked as an HPX action (by a HPX future)
typedef plain_result_action1<std::size_t, std::size_t, hello_world_worker>
    hello_world_worker_action;

HPX_REGISTER_PLAIN_ACTION(hello_world_worker_action);

typedef eager_future<hello_world_worker_action> hello_world_worker_future;

///////////////////////////////////////////////////////////////////////////////
void hello_world_foreman()
{
    std::size_t const shepherds = get_num_os_threads();
    id_type const prefix = find_here();

    std::set<std::size_t> attendance;
    for (std::size_t shepherd = 0; shepherd < shepherds; ++shepherd)
        attendance.insert(shepherd);

    while (!attendance.empty())
    {
        std::vector<future_value<std::size_t> > futures;
        BOOST_FOREACH(std::size_t shepherd, attendance)
        {
            futures.push_back(hello_world_worker_future(prefix, shepherd)); 
        }

        // wait for all of the futures to return their values, we re-spawn the
        // future until the action gets executed on the right OS-thread
        wait(futures, [&](std::size_t, std::size_t t) { attendance.erase(t); });
    }
}

// Define the boilerplate code necessary for the function 'hello_world_foreman'
// to be invoked as an HPX action
typedef plain_action0<hello_world_foreman> hello_world_foreman_action;

HPX_REGISTER_PLAIN_ACTION(hello_world_foreman_action);

typedef eager_future<hello_world_foreman_action> hello_world_foreman_future;

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map&)
{
    {
        std::vector<id_type> prefixes = find_all_localities();

        std::vector<future_value<void> > futures;
        BOOST_FOREACH(id_type const& node, prefixes)
        { 
            futures.push_back(hello_world_foreman_future(node)); 
        }

        wait(futures);    // Wait for all IO to finish
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
       desc_commandline("usage: " HPX_APPLICATION_STRING " [options]");

    // Initialize and run HPX.
    return init(desc_commandline, argc, argv);
}

