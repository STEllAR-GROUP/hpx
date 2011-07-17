//  Copyright (c)      2011 Bryce Lelbach
//  Copyright (c) 2009-2010 Dylan Stark
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#include <boost/format.hpp>
#include <boost/cstdint.hpp>

#include <hpx/hpx_init.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/applier/apply.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/include/performance_counters.hpp>
#include <hpx/state.hpp>
#include <hpx/util/high_resolution_timer.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/components/plain_component_factory.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using boost::format;

using hpx::cout;
using hpx::flush;

using hpx::init;
using hpx::finalize;
using hpx::running;
using hpx::find_here;

using hpx::lcos::eager_future;

using hpx::applier::get_applier;
using hpx::applier::apply;

using hpx::actions::plain_action1;
using hpx::actions::plain_action4;

using hpx::threads::threadmanager_is;

using hpx::performance_counters::stubs::performance_counter;
using hpx::performance_counters::counter_value;
using hpx::performance_counters::status_valid_data;

using hpx::naming::gid_type;

using hpx::util::high_resolution_timer;

///////////////////////////////////////////////////////////////////////////////
void monitor(
    std::string const& name
  , double frequency
  , double duration
  , double rate
) {
    // Resolve the GID of the performance counter using it's symbolic name.
    gid_type gid;
    get_applier().get_agas_client().queryid(name, gid);

    BOOST_ASSERT(gid); 

    high_resolution_timer t;
    double current_time(0);

    for (boost::uint64_t segment = 0; true; ++segment)
    {
        // Get the current time at the start of each segment.
        const double segment_start = t.elapsed();

        // do-while style for loop.
        for (boost::uint64_t block = 0;
             block != 0 || current_time - segment_start < frequency;
             ++block)
        {
            // Start the monitoring phase.
            const double monitor_start = t.elapsed();

            do {
                current_time = t.elapsed();

                // Query the performance counter.
                counter_value value = performance_counter::get_value(gid); 

                if (HPX_LIKELY(status_valid_data == value.status_))
                    cout() << (format("(%s %f %#x %#x %#x)\n")
                              % name
                              % current_time
                              % segment
                              % block
                              % value.value_) << flush;
                else
                    cout() << (format("(%s %f %#x %#x '())\n")
                              % name
                              % current_time
                              % segment
                              % block
                              % value.value_) << flush;

                // Adjust rate of pinging values.
                const double delay_start = t.elapsed();

                do {
                    // Stop when the threadmanager is no longer available.
                    if (!threadmanager_is(running))
                        return;

                    current_time = t.elapsed();
                } while (current_time - delay_start < rate);
            } while (current_time - monitor_start < duration);
        }

        // Adjust rate of monitoring phases.
        const double pause_start = t.elapsed();

        do {
            current_time = t.elapsed();
        } while (current_time - pause_start < (frequency - duration));
    }
}

typedef plain_action4<
    // arguments
    std::string const&
  , double
  , double
  , double
    // function
  , monitor
> monitor_action;

HPX_REGISTER_PLAIN_ACTION(monitor_action);

typedef eager_future<monitor_action> monitor_future;

///////////////////////////////////////////////////////////////////////////////
void fork_bomb(boost::uint64_t count);

typedef plain_action1<
    // arguments
    boost::uint64_t
    // function
  , fork_bomb
> fork_bomb_action;

HPX_REGISTER_PLAIN_ACTION(fork_bomb_action);

void fork_bomb(boost::uint64_t count)
{
    for (boost::uint64_t i = 0; i < count; ++i)
        apply<fork_bomb_action>(find_here(), count);        
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm)
{
    {
        const std::string name = vm["name"].as<std::string>();
        const double frequency = vm["frequency"].as<double>(); 
        const double duration = vm["duration"].as<double>(); 
        const double rate = vm["rate"].as<double>();

        monitor_future mf(find_here(), name, frequency, duration, rate);

        // Create a TON of work recursively.
        fork_bomb(4); 
 
        mf.get(); 
    }

    finalize();
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options.
    options_description
       desc_commandline("usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ( "name"
        , value<std::string>()->default_value
            ("/queue([L1]/threadmanager)/length")
        , "symbolic name of the performance counter")

        ( "frequency"
        , value<double>()->default_value(10, "10") 
        , "frequency of monitoring")

        ( "duration"
        , value<double>()->default_value(5, "5") 
        , "duration of each monitoring block")

        ( "rate"
        , value<double>()->default_value(0.01, "0.01") 
        , "rate of polling")
        ;

    // Initialize and run HPX.
    return init(desc_commandline, argc, argv);
}

