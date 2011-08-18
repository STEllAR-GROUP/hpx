////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach and Katelyn Kufahl
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <unistd.h>

#include <vector>
#include <list>
#include <iostream>

#include <boost/cstdint.hpp>
#include <boost/format.hpp>
#include <boost/foreach.hpp>

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/components/plain_component_factory.hpp>
#include <hpx/util/high_resolution_timer.hpp>
#include <hpx/lcos/eager_future.hpp>
#include <hpx/include/iostreams.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using hpx::naming::gid_type;
using hpx::naming::get_prefix_from_gid;

using hpx::applier::applier;
using hpx::applier::get_applier;

using hpx::actions::plain_action0;

using hpx::lcos::eager_future;
using hpx::lcos::future_value;

using hpx::util::high_resolution_timer;

using hpx::get_runtime;
using hpx::init;
using hpx::finalize;
using hpx::cout;
using hpx::endl;

///////////////////////////////////////////////////////////////////////////////
void noop() {}

typedef plain_action0<noop> noop_action;

HPX_REGISTER_PLAIN_ACTION(noop_action);

typedef eager_future<noop_action> noop_future;

///////////////////////////////////////////////////////////////////////////////
void print_count()
{
    applier& a = get_applier();

    char hostname[128];

    gethostname(hostname, sizeof(hostname));

    cout() << (boost::format("locality %1% (%2%, %3% shepherds) start sending "
                             "%4% parcels and completed sending %5% parcels")
              % a.get_prefix_id()
              % hostname
              % get_runtime().get_process().get_num_os_threads()
              % a.get_parcel_handler().get_parcelport().total_sends_started()
              % a.get_parcel_handler().get_parcelport().total_sends_completed())
           << endl;
}

typedef plain_action0<print_count> print_count_action;

HPX_REGISTER_PLAIN_ACTION(print_count_action);

typedef eager_future<print_count_action> print_count_future;

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm)
{
    {
        boost::uint64_t count = vm["count"].as<boost::uint64_t>();

        std::vector<gid_type> localities;
        get_applier().get_agas_client().get_prefixes(localities);

        std::list<future_value<void> > futures;

        BOOST_FOREACH(gid_type const& node, localities)
        {
            cout() << (boost::format("starting %1% futures on locality %2%")
                      % count % get_prefix_from_gid(node))
                   << endl;
            for (boost::uint64_t i = 0; i < count; ++i)
                futures.push_back(noop_future(node));
        }

        BOOST_FOREACH(future_value<void> const& f, futures)
        { f.get(); } 

        futures.clear();

        BOOST_FOREACH(gid_type const& node, localities)
        { futures.push_back(print_count_future(node)); }

        BOOST_FOREACH(future_value<void> const& f, futures)
        { f.get(); } 
    }

    finalize();

    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    options_description
       desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ( "count"
        , value<boost::uint64_t>()->default_value(8192)
        , "number of futures to create per locality") 
        ;

    // Initialize and run HPX
    return init(desc_commandline, argc, argv);
}

