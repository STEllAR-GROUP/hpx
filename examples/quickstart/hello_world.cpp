///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach 
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#include <vector>
#include <list>
#include <set>

#include <boost/ref.hpp>
#include <boost/foreach.hpp>
#include <boost/format.hpp>

#include <hpx/hpx_init.hpp>
#include <hpx/lcos/eager_future.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/components/plain_component_factory.hpp>
#include <hpx/include/iostreams.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;

using boost::format;

using hpx::init;
using hpx::finalize;

using hpx::naming::id_type;
using hpx::naming::get_gid_from_prefix;

using hpx::get_runtime;

using hpx::applier::get_prefix_id;
using hpx::applier::get_applier;
using hpx::applier::register_work;

using hpx::actions::plain_action0;
using hpx::actions::plain_result_action1;

using hpx::lcos::future_value;
using hpx::lcos::eager_future;

using hpx::threads::threadmanager_base;

using hpx::cout;
using hpx::endl;

///////////////////////////////////////////////////////////////////////////////
std::size_t hello_world_worker(std::size_t desired)
{
    std::size_t current = threadmanager_base::get_thread_num();

    if (current == desired)
    {
        cout() << ( format("hello world from shepherd %1% on locality %2%")
                  % desired 
                  % get_prefix_id())
            << endl;
        return desired;
    }

    else
        return std::size_t(-1);
}

typedef plain_result_action1<std::size_t, std::size_t, hello_world_worker>
    hello_world_worker_action;

HPX_REGISTER_PLAIN_ACTION(hello_world_worker_action);

typedef eager_future<hello_world_worker_action> hello_world_worker_future;

///////////////////////////////////////////////////////////////////////////////
void hello_world_foreman()
{
    const std::size_t shepherds
        = get_runtime().get_config().get_num_shepherds();

    const id_type prefix = get_applier().get_runtime_support_gid();

    std::set<std::size_t> attendence;

    for (std::size_t shepherd = 0; shepherd < shepherds; ++shepherd)
        attendence.insert(shepherd);

    while (!attendence.empty())
    {
        std::list<future_value<std::size_t> > futures;

        BOOST_FOREACH(std::size_t shepherd, attendence)
        { futures.push_back(hello_world_worker_future(prefix, shepherd)); }

        BOOST_FOREACH(future_value<std::size_t> const& f, futures)
        {
            std::size_t r = f.get();
            if (r != std::size_t(-1))
                attendence.erase(r);
        } 
    }
}

typedef plain_action0<hello_world_foreman> hello_world_foreman_action;

HPX_REGISTER_PLAIN_ACTION(hello_world_foreman_action);

typedef eager_future<hello_world_foreman_action> hello_world_foreman_future;

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map&)
{
    {
        std::list<future_value<void> > futures;
        std::vector<id_type> prefixes;

        get_applier().get_prefixes(prefixes);

        BOOST_FOREACH(id_type const& node, prefixes)
        { futures.push_back(hello_world_foreman_future(node)); }

        // Wait for all IO to finish
        BOOST_FOREACH(future_value<void> const& f, futures)
        { f.get(); } 
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

    // Initialize and run HPX.
    return init(desc_commandline, argc, argv);
}

