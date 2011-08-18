////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011      Bryce Lelbach
//  Copyright (c) 2009-2010 Dylan Stark
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/hpx_fwd.hpp>

#include <iomanip>
#include <cmath>
#include <cfloat>

#include <boost/ref.hpp>
#include <boost/bind.hpp>
#include <boost/atomic.hpp>
#include <boost/cstdint.hpp>
#include <boost/foreach.hpp>
#include <boost/format.hpp>
#include <boost/integer_traits.hpp>
#include <boost/thread.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

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
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/state.hpp>
#include <hpx/util/high_resolution_timer.hpp>

#include <examples/math/distributed/discovery/discovery.hpp>
#include <examples/math/distributed/integrator/integrator.hpp>

using boost::format;

using boost::program_options::options_description;
using boost::program_options::value;
using boost::program_options::variables_map;

using boost::posix_time::milliseconds;

using hpx::actions::function;
using hpx::actions::plain_action1;
using hpx::actions::plain_action4;
using hpx::actions::plain_result_action1;

using hpx::applier::register_work;
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
using hpx::network_error;
using hpx::running;

using hpx::components::get_component_type;

using hpx::lcos::eager_future;
using hpx::lcos::future_value;
using hpx::lcos::base_lco;

using hpx::naming::resolver_client;
using hpx::naming::get_prefix_from_gid;
using hpx::naming::get_prefix_from_id;
using hpx::naming::get_id_from_prefix;
using hpx::naming::gid_type;
using hpx::naming::id_type;

using hpx::performance_counters::counter_value;
using hpx::performance_counters::status_valid_data;
using hpx::performance_counters::stubs::performance_counter;

using hpx::threads::threadmanager_is;
using hpx::threads::thread_priority_critical;
using hpx::threads::wait_timeout;
using hpx::threads::pending;
using hpx::threads::suspended;
using hpx::threads::get_self_id;
using hpx::threads::get_self;
using hpx::threads::set_thread_state;

using hpx::util::high_resolution_timer;

using std::abs;
using std::log;
using std::pow;

struct call_tag {};

///////////////////////////////////////////////////////////////////////////////
double math_function (double const& r)
{
    return abs(sin(pow(r, 0.25)) / (log(r) * log(r))); 
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
void agas_main(variables_map& vm)
{
    const boost::uint32_t rate = vm["monitoring-rate"].as<boost::uint32_t>();
    const std::string name = vm["monitor"].as<std::string>();

    std::cout << (format("monitoring %s\n") % name);

    // Resolve the GID of the performance counter using it's symbolic name.
    gid_type gid;
    get_applier().get_agas_client().queryid(name, gid);

    BOOST_ASSERT(gid); 

    future_value<void> stop_flag;

    // Associate the stop flag with a symbolic name.
    get_applier().get_agas_client().registerid
        ("/stop_flag([L1]/solver_double)", stop_flag.get_gid().get_gid());

    boost::int64_t zero_time = 0;

    while (true) 
    {
        if (!threadmanager_is(running) || stop_flag.ready())
            return;

        // Query the performance counter.
        counter_value value = performance_counter::get_value(gid); 

        if (HPX_LIKELY(status_valid_data == value.status_))
        {
            if (!zero_time)
                zero_time = value.time_;

            std::cout << ( format("  %d,%d\n")
                         % (value.time_ - zero_time)
                         % value.value_);
        }

        // Live wait.
        //boost::this_thread::sleep(boost::get_system_time() + 
        //    boost::posix_time::milliseconds(rate));

        // Schedule a wakeup.
        set_thread_state(get_self_id(), milliseconds(rate)
                       , pending, wait_timeout, thread_priority_critical);
        
        get_self().yield(suspended);
    }
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

    if (!vm.count("disable-monitoring"))
    {
        // Kill the monitor.
        resolver_client& agas_client = get_runtime().get_agas_client();
        gid_type stop_flag_gid;
    
        if (!agas_client.queryid("/stop_flag([L1]/solver_double)", stop_flag_gid))
        {
            HPX_THROW_EXCEPTION(network_error, "console_main",
                "couldn't find stop flag");
        } 
    
        BOOST_ASSERT(stop_flag_gid);
    
        eager_future<base_lco::set_event_action> stop_future(stop_flag_gid);
    
        stop_future.get();
    }

    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm)
{
    int r = 0;

    {
        if (get_prefix_id() == 1 && !vm.count("disable-monitoring"))
        {
            if (get_runtime().get_agas_client().is_console())
            {
                register_work(boost::bind(&agas_main, boost::ref(vm)));
                r = console_main(vm); 
            }
            else
                agas_main(vm); 
        }
        else
            r = console_main(vm);
    }

    finalize();
    return r;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    options_description
       desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

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

        ( "monitor"
        , value<std::string>()->default_value
            ("/pcs/time([L1]/threadmanager)/maintenance")
        , "symbolic name of the performance counter to monitor on the AGAS "
          "locality")

        ( "monitoring-rate"
        , value<boost::uint32_t>()->default_value(100) 
        , "milliseconds between each performance counter query")

        ( "disable-monitoring"
        , "turn off performance counter monitoring")
        ;

    // Initialize and run HPX
    return init(desc_commandline, argc, argv);
}

