////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <iomanip>
#include <cmath>

#include <boost/cstdint.hpp>
#include <boost/format.hpp>
#include <boost/foreach.hpp>
#include <boost/rational.hpp>
#include <boost/integer_traits.hpp>

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/lcos/eager_future.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/components/component_factory.hpp>
#include <hpx/runtime/components/plain_component_factory.hpp>
#include <hpx/util/serialize_rational.hpp>
#include <hpx/util/high_resolution_timer.hpp>
#include <examples/math/distributed/discovery/discovery.hpp>
#include <examples/math/distributed/integrator/integrator.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using hpx::naming::id_type;
using hpx::naming::gid_type;
using hpx::naming::get_prefix_from_gid;

using hpx::applier::get_applier;

using hpx::components::get_component_type;

using hpx::actions::plain_result_action1;
using hpx::actions::function;

using hpx::lcos::eager_future;
using hpx::lcos::future_value;

using hpx::balancing::discovery;
using hpx::balancing::topology_map;
using hpx::balancing::integrator;

using hpx::get_runtime;
using hpx::init;
using hpx::finalize;
using hpx::find_here;

using hpx::cout;
using hpx::cerr;
using hpx::flush;
using hpx::endl;

using hpx::util::high_resolution_timer;

///////////////////////////////////////////////////////////////////////////////
typedef boost::rational<boost::int64_t> rational64;

///////////////////////////////////////////////////////////////////////////////
rational64 math_function (rational64 const& r)
{
    return rational64(sqrt(r.numerator()), sqrt(r.denominator())); 
}

typedef plain_result_action1<
    // result type
    rational64 
    // arguments
  , rational64 const&  
    // function
  , math_function
> math_function_action;

HPX_REGISTER_PLAIN_ACTION(math_function_action);

///////////////////////////////////////////////////////////////////////////////
int master(variables_map& vm)
{
    // Get options.
    const rational64 lower_bound = vm["lower-bound"].as<rational64>();
    const rational64 upper_bound = vm["upper-bound"].as<rational64>();
    const rational64 tolerance = vm["tolerance"].as<rational64>();

    const boost::uint64_t top_segs
        = vm["top-segments"].as<boost::uint64_t>();

    const boost::uint64_t regrid_segs
        = vm["regrid-segments"].as<boost::uint64_t>();

    // Handle for the root discovery component. 
    discovery disc_root;

    // Create the first discovery component on this locality.
    disc_root.create(find_here());

    cout << "deploying discovery infrastructure" << endl;

    // Deploy the scheduling infrastructure.
    std::vector<id_type> discovery_network = disc_root.build_network_sync(); 

    cout << ( boost::format("root discovery server is at %1%")
              % disc_root.get_gid())
         << endl;

    // Get this localities topology map LVA.
    topology_map* topology_ptr
        = reinterpret_cast<topology_map*>(disc_root.topology_lva_sync());

    topology_map& topology = *topology_ptr;

    // Print out the system topology.
    boost::uint32_t total_shepherds = 0;
    for (std::size_t i = 1; i <= topology.size(); ++i)
    {
        cout << ( boost::format("locality %1% has %2% shepherds")
                  % i 
                  % topology[i])
             << endl;
        total_shepherds += topology[i]; 
    }

    cout << ( boost::format("%1% localities, %2% shepherds total")
              % topology.size()
              % total_shepherds)
         << endl;

    // Create the function that we're integrating.
    function<rational64(rational64 const&)> f(new math_function_action);

    // Handle for the root integrator component. 
    integrator<rational64> integ_root;

    // Create the initial integrator component on this locality. 
    integ_root.create(find_here());

    cout << "deploying integration infrastructure" << endl;

    const rational64 eps(1, boost::integer_traits<boost::int64_t>::const_max);

    // Now, build the integration infrastructure on all nodes.
    std::vector<id_type> integrator_network =
        integ_root.build_network_sync
            (discovery_network, f, tolerance, regrid_segs, eps); 

    cout << ( boost::format("root integration server is at %1%")
              % integ_root.get_gid())
         << endl;

    // Print out the GIDs of the discovery and integrator servers.
    for (std::size_t i = 0; i < integrator_network.size(); ++i)
    {
        // These vectors are sorted from lowest prefix to highest prefix.
        cout << ( boost::format("locality %1% infrastructure\n"
                                  "  discovery server at %2%\n" 
                                  "  integration server at %3%")
                  % (i + 1)
                  % discovery_network[i]
                  % integrator_network[i])
             << endl; 
    }

    // Start the timer.
    high_resolution_timer t;

    // Solve the integral using an adaptive trapezoid algorithm.
    rational64 r = integ_root.solve_sync(lower_bound, upper_bound, top_segs);

    double elapsed = t.elapsed();

    cout << ( boost::format("integral from %1% to %2% is %3%\n"
                              "computation took %4% seconds")
              % boost::rational_cast<long double>(lower_bound)
              % boost::rational_cast<long double>(upper_bound)
              % boost::rational_cast<long double>(r)
              % elapsed)
         << endl;

    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm)
{
    int r = master(vm);
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
        , value<rational64>()->default_value(rational64(0), "0") 
        , "lower bound of integration")

        ( "upper-bound"
        , value<rational64>()->default_value(rational64(128), "128")
        , "upper bound of integration")

        ( "tolerance"
        , value<rational64>()->default_value(rational64(0, 10), "0.1") 
        , "resolution tolerance")

        ( "top-segments"
        , value<boost::uint64_t>()->default_value(4096) 
        , "number of top-level segments")

        ( "regrid-segments"
        , value<boost::uint64_t>()->default_value(128) 
        , "number of segment per regrid")
        ;

    // Initialize and run HPX
    return init(desc_commandline, argc, argv);
}

