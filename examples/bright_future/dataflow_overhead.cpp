
//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/include/client.hpp>

#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/util/high_resolution_timer.hpp>

#include <hpx/components/dataflow/dataflow.hpp>
#include <hpx/components/dataflow/async_dataflow_wait.hpp>

#include <utility>

using hpx::cout;
using hpx::flush;

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using hpx::init;
using hpx::finalize;
using hpx::find_here;
using hpx::find_all_localities;
using hpx::lcos::dataflow;
using hpx::lcos::dataflow_base;
using hpx::lcos::wait;
using hpx::naming::id_type;
using hpx::util::high_resolution_timer;


using hpx::actions::plain_result_action0;

///////////////////////////////////////////////////////////////////////////////
// we use globals here to prevent the delay from being optimized away
double global_scratch = 0;
boost::uint64_t num_iterations = 0;

///////////////////////////////////////////////////////////////////////////////
double null_function()
{
    double d = 0.0;
    for (double i = 0; i < num_iterations; ++i)
        d += 1 / (2. * i + 1);
    return d;
}

HPX_PLAIN_ACTION(null_function, null_action)

typedef dataflow<null_action> null_dataflow;

int hpx_main(variables_map & vm)
{
    {
        num_iterations = vm["delay-iterations"].as<boost::uint64_t>();

        const boost::uint64_t count = vm["dataflows"].as<boost::uint64_t>();
        
        std::vector<id_type> prefixes = find_all_localities();
        
        double function_time;
        {    
            high_resolution_timer walltime;
            for(std::size_t i = 0; i < 10000; ++i)
            {
                global_scratch += null_function();
            }
            function_time = walltime.elapsed()/10000;
        }

        BOOST_FOREACH(id_type const & prefix, prefixes)
        {
            std::vector<dataflow_base<double> > dataflows;
            dataflows.reserve(count);

            high_resolution_timer walltime;

            for(boost::uint64_t i = 0; i < count; ++i)
            {
                dataflows.push_back(null_dataflow(prefix));
            }

            wait(dataflows, [&] (std::size_t, double r) { global_scratch += r;});
            const double duration = walltime.elapsed();
            
            if (vm.count("csv"))
                cout << ( boost::format("%1%,%2%,%3%,%4%\n")
                        % get_locality_id_from_id(prefix)
                        % count
                        % duration
                        % function_time)
                     << flush;
            else
                cout << ( boost::format("Locality %1%: invoked %2% futures in %3% seconds (average workload %4% s)\n")
                        % get_locality_id_from_id(prefix)
                        % count
                        % duration
                        % function_time)
                     << flush;
        }
    }
    finalize();
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(
    int argc
  , char* argv[]
    )
{
    // Configure application-specific options.
    options_description cmdline("usage: " HPX_APPLICATION_STRING " [options]");

    cmdline.add_options()
        ( "dataflows"
        , value<boost::uint64_t>()->default_value(500000)
        , "number of dataflows to invoke")

        ( "delay-iterations"
        , value<boost::uint64_t>()->default_value(0)
        , "number of iterations in the delay loop")

        ( "csv"
        , "output results as csv (format: locality,count,duration)")
        ;

    // Initialize and run HPX.
    return init(cmdline, argc, argv);
}
