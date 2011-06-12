///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach 
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#include <vector>
#include <list>
#include <iostream>

#include <boost/ref.hpp>
#include <boost/foreach.hpp>

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/lcos/eager_future.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/components/plain_component_factory.hpp>
#include <hpx/runtime/components/server/manage_component.hpp>
#include <hpx/components/iostreams/lazy_ostream.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;

using hpx::init;
using hpx::finalize;

using hpx::naming::id_type;
using hpx::naming::get_gid_from_prefix;

using hpx::applier::get_prefix_id;
using hpx::applier::get_applier;

using hpx::actions::plain_action1;

using hpx::lcos::future_value;
using hpx::lcos::eager_future;

using hpx::components::server::create_one;
using hpx::components::managed_component;

using hpx::endl;
using hpx::iostreams::lazy_ostream;
using hpx::iostreams::server::output_stream;

typedef managed_component<output_stream> ostream_type;

void hello_world(id_type cout_gid)
{
    lazy_ostream hpx_cout(cout_gid);
    hpx_cout << "[L" << get_prefix_id() << "]: hello world!" << endl;
}

typedef plain_action1<id_type, hello_world> hello_world_action;

HPX_REGISTER_PLAIN_ACTION(hello_world_action);

typedef eager_future<hello_world_action> hello_world_future;

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
        { futures.push_back(hello_world_future(node, cout_gid)); }

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
       desc_commandline("usage: " HPX_APPLICATION_STRING " [options]");

    // Initialize and run HPX.
    return init(desc_commandline, argc, argv);
}

