///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach 
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#include <vector>
#include <list>

#include <boost/ref.hpp>
#include <boost/foreach.hpp>
#include <boost/format.hpp>

#include <hpx/hpx_init.hpp>
#include <hpx/lcos/eager_future.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/components/plain_component_factory.hpp>
#include <hpx/components/iostreams/lazy_ostream.hpp>

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

using hpx::actions::plain_action1;
using hpx::actions::plain_result_action2;

using hpx::lcos::future_value;
using hpx::lcos::eager_future;

using hpx::threads::threadmanager_base;

using hpx::components::server::create_one;
using hpx::components::managed_component;

using hpx::endl;
using hpx::iostreams::lazy_ostream;
using hpx::iostreams::server::output_stream;

typedef managed_component<output_stream> ostream_type;

///////////////////////////////////////////////////////////////////////////////
bool hello_world_worker(id_type const& cout_gid, std::size_t desired)
{
    std::size_t current = threadmanager_base::get_thread_num();

    if (current == desired)
    {
        lazy_ostream hpx_cout(cout_gid);
        hpx_cout << boost::str(
                        format("hello world from shepherd %1% on locality %2%")
                      % desired 
                      % get_prefix_id())
            << endl;
        return true;
    }

    else
        return false;
}

typedef plain_result_action2<
    bool, id_type const&, std::size_t, hello_world_worker
> hello_world_worker_action;

HPX_REGISTER_PLAIN_ACTION(hello_world_worker_action);

typedef eager_future<hello_world_worker_action> hello_world_worker_future;

///////////////////////////////////////////////////////////////////////////////
void hello_world_foreman(id_type const& cout_gid)
{
    const std::size_t shepherds
        = get_runtime().get_config().get_num_os_threads();

    const id_type prefix = get_applier().get_runtime_support_gid();

    for (std::size_t i = 0; i < shepherds; ++i)
    {
        while (true)
        {
            hello_world_worker_future f(prefix, cout_gid, i);
            
            if (f.get())
                break;
        }
    }
}

typedef plain_action1<id_type const&, hello_world_foreman>
    hello_world_foreman_action;

HPX_REGISTER_PLAIN_ACTION(hello_world_foreman_action);

typedef eager_future<hello_world_foreman_action> hello_world_foreman_future;

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map&)
{
    {
        id_type cout_gid(create_one<ostream_type>(boost::ref(std::cout))
                       , id_type::managed);

        lazy_ostream hpx_cout(cout_gid);

        std::list<future_value<void> > futures;
        std::vector<id_type> prefixes;

        get_applier().get_prefixes(prefixes);

        BOOST_FOREACH(id_type const& node, prefixes)
        { futures.push_back(hello_world_foreman_future(node, cout_gid)); }

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

