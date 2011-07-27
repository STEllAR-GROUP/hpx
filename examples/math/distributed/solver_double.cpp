////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011      Bryce Lelbach
//  Copyright (c) 2009-2010 Dylan Stark
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <iomanip>
#include <cmath>
#include <cfloat>

#include <boost/atomic.hpp>
#include <boost/cstdint.hpp>
#include <boost/foreach.hpp>
#include <boost/format.hpp>
#include <boost/integer_traits.hpp>

#include <hpx/exception.hpp>
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/include/performance_counters.hpp>
#include <hpx/lcos/eager_future.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/components/component_factory.hpp>
#include <hpx/runtime/components/plain_component_factory.hpp>
#include <hpx/state.hpp>
#include <hpx/util/high_resolution_timer.hpp>

#include <examples/math/distributed/discovery/discovery.hpp>
#include <examples/math/distributed/integrator/integrator.hpp>

using boost::format;

using boost::program_options::options_description;
using boost::program_options::value;
using boost::program_options::variables_map;

using hpx::actions::function;
using hpx::actions::plain_action1;
using hpx::actions::plain_action4;
using hpx::actions::plain_result_action1;

using hpx::applier::get_applier;
using hpx::applier::get_prefix_id;

using hpx::balancing::discovery;
using hpx::balancing::integrator;
using hpx::balancing::topology_map;

using hpx::cerr;
using hpx::cout;
using hpx::endl;
using hpx::flush;

using hpx::init;
using hpx::finalize;
using hpx::find_here;
using hpx::get_runtime;
using hpx::running;
using hpx::runtime_mode_probe;

using hpx::components::get_component_type;

using hpx::lcos::eager_future;
using hpx::lcos::future_value;
using hpx::naming::get_prefix_from_gid;

using hpx::naming::get_prefix_from_id;
using hpx::naming::gid_type;
using hpx::naming::id_type;

using hpx::performance_counters::counter_value;
using hpx::performance_counters::status_valid_data;
using hpx::performance_counters::stubs::performance_counter;

using hpx::threads::threadmanager_is;

using hpx::util::high_resolution_timer;

using std::abs;
using std::log;
using std::pow;

struct call_tag {};

///////////////////////////////////////////////////////////////////////////////
double math_function (double const& r)
{
    return abs(sin(pow(r, 0.25L)) / (log(r) * log(r))); 
}

typedef plain_result_action1<
    // result type
    double 
    // arguments
  , double const&  
    // function
  , math_function
> math_function_action;

HPX_REGISTER_PLAIN_ACTION(math_function_action);

///////////////////////////////////////////////////////////////////////////////
int agas_main(variables_map& vm)
{
    const double frequency = 10;
    const double duration = 5; 
    const double rate = 0.01;

    const std::string name = vm["performance-counter"].as<std::string>();

    std::cout << (format("(monitor %s)\n") % name);

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

                if (HPX_LIKELY(status_valid_data == value.status_)) {
                    std::cout << ( format("  (%f %d)\n")
                                 % current_time
                                 % value.value_);
                }
                else {
                    std::cout << (format("  (%f '())\n") % current_time);
                }

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

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options.
    options_description
       desc_commandline("usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ;

    // Initialize and run HPX, enforce probe mode as we connect to an existing 
    // application.
#if defined(BOOST_WINDOWS)
    return init(desc_commandline, argc, argv, install_windows_counters, 
        uninstall_windows_counters, runtime_mode_probe);
#else
    return init(desc_commandline, argc, argv, runtime_mode_probe);
#endif
}


///////////////////////////////////////////////////////////////////////////////
int console_main(variables_map& vm)
{
    // Get options.
    const double lower_bound = vm["lower-bound"].as<double>();
    const double upper_bound = vm["upper-bound"].as<double>();
    const double tolerance = vm["tolerance"].as<double>();

    const boost::uint32_t top_segs
        = vm["top-segments"].as<boost::uint32_t>();

    const boost::uint32_t regrid_segs
        = vm["regrid-segments"].as<boost::uint32_t>();

    // Handle for the root discovery component. 
    discovery disc_root;

    // Create the first discovery component on this locality.
    disc_root.create(find_here());

    cout() << "deploying discovery infrastructure" << endl;

    // Deploy the scheduling infrastructure.
    std::vector<id_type> discovery_network = disc_root.build_network_sync(); 

    cout() << ( format("root discovery server is at %1%")
              % disc_root.get_gid())
           << endl;

    // Get this localities topology map LVA.
    topology_map* topology_ptr
        = reinterpret_cast<topology_map*>(disc_root.topology_lva_sync());

    topology_map& topology = *topology_ptr;

    // Print out the system topology.
    boost::uint32_t total_shepherds = 0;
    BOOST_FOREACH(topology_map::value_type const& kv, topology)
    {
        cout() << ( format("locality %1% has %2% shepherds")
                  % kv.first 
                  % kv.second)
               << endl;
        total_shepherds += kv.second; 
    }

    cout() << ( format("%1% localities, %2% shepherds total")
              % topology.size()
              % total_shepherds)
           << endl;

    // Create the function that we're integrating.
    function<double(double const&)> f(new math_function_action);

    // Handle for the root integrator component. 
    integrator<double> integ_root;

    // Create the initial integrator component on this locality. 
    integ_root.create(find_here());

    cout() << "deploying integration infrastructure" << endl;

    const double eps(DBL_EPSILON);

    // Now, build the integration infrastructure on all nodes.
    std::vector<id_type> integrator_network =
        integ_root.build_network_sync
            (discovery_network, f, tolerance, regrid_segs, eps); 

    cout() << ( format("root integration server is at %1%")
              % integ_root.get_gid())
           << endl;

    // Print out the GIDs of the discovery and integrator servers.
    for (std::size_t i = 0; i < integrator_network.size(); ++i)
    {
        // These vectors are sorted from lowest prefix to highest prefix.
        cout() << ( format("locality %1% infrastructure\n"
                                  "  discovery server at %2%\n" 
                                  "  integration server at %3%")
                  % get_prefix_from_id(discovery_network[i])
                  % discovery_network[i]
                  % integrator_network[i])
               << endl; 
    }

    // Start the timer.
    high_resolution_timer t;

    // Solve the integral using an adaptive trapezoid algorithm.
    double r = integ_root.solve_sync(lower_bound, upper_bound, top_segs);

    double elapsed = t.elapsed();

    cout() << ( format("integral from %.12f to %.12f is %.12f\n"
                              "computation took %f seconds")
              % lower_bound
              % upper_bound
              % r
              % elapsed)
           << endl;

    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm)
{
    if (get_prefix_id() == 1)
        int r = agas_main(vm); 
    else
        int r = console_main(vm);

    finalize();
    return r;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    options_description
       desc_commandline("usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ( "lower-bound"
        , value<double>()->default_value(1 << 13, "2^13") 
        , "lower bound of integration")

        ( "upper-bound"
        , value<double>()->default_value(1 << 25, "2^25")
        , "upper bound of integration")

        ( "tolerance"
        , value<double>()->default_value(1e-7, "1e-7") 
        , "resolution tolerance")

        ( "top-segments"
        , value<boost::uint32_t>()->default_value(1 << 25, "2^25") 
        , "number of top-level segments")

        ( "regrid-segments"
        , value<boost::uint32_t>()->default_value(128) 
        , "number of segment per regrid")

        ( "performance-counter"
        , value<std::string>()->default_value
            ("/time([L1]/threadmanager)/maintenance")
        , "symbolic name of the performance counter to monitor on the AGAS "
          "locality")
        ;

    // Initialize and run HPX
    return init(desc_commandline, argc, argv);
}

