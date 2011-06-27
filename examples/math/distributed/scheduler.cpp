////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <iomanip>

#include <boost/cstdint.hpp>
#include <boost/format.hpp>
#include <boost/foreach.hpp>

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/components/plain_component_factory.hpp>
#include <hpx/lcos/eager_future.hpp>
#include <hpx/include/iostreams.hpp>

#include <examples/math/distributed/discovery/discovery.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using hpx::naming::id_type;
using hpx::naming::gid_type;
using hpx::naming::get_prefix_from_gid;

using hpx::applier::get_applier;

using hpx::components::get_component_type;

using hpx::lcos::eager_future;
using hpx::lcos::future_value;

using hpx::balancing::discovery;
using hpx::balancing::topology_map;

using hpx::get_runtime;
using hpx::init;
using hpx::finalize;
using hpx::find_here;

using hpx::cout;
using hpx::cerr;
using hpx::flush;

///////////////////////////////////////////////////////////////////////////////
inline bool equal(double a, double b)
{ return std::fabs(a - b) < 0.001; }

///////////////////////////////////////////////////////////////////////////////
int master(variables_map& vm)
{
    cout() << std::setprecision(12);
    cerr() << std::setprecision(12);

    const std::size_t points = vm["points"].as<std::size_t>();

    // {{{ machine discovery  
    // Handle for the root discovery component. 
    discovery root;

    // Create the first discovery component on this locality.
    root.create(find_here());

    cout() << "deploying discovery infrastructure\n";

    // Deploy the scheduling infrastructure.
    std::vector<id_type> network = root.deploy_sync(); 

    // Get this localities topology map LVA
    topology_map* topology_ptr
        = reinterpret_cast<topology_map*>(root.topology_lva());

    topology_map& topology = *topology_ptr;

    std::size_t total_shepherds = 0;
    BOOST_FOREACH(topology_map::value_type const& v, topology)
    {
        cout() << ( boost::format("locality %1% has %2% shepherds\n")
                  % get_prefix_from_gid(v.first)
                  % v.second);
        total_shepherds += v.second;
    }

    std::vector<gid_type> localities;
    get_applier().get_agas_client().get_prefixes
        (localities, get_component_type<hpx::balancing::server::discovery>());

    if (topologies.size() != localities.size())
    {
        cout() << flush;
        cerr() << "error: AGAS bug, mismatched locality lists encountered"
               << endl; 
        return 1;
    } 
    
    cout() << ( boost::format("%1% localities, %2% shepherds total\n")
              % topology.size()
              % total_shepherds);
    // }}}

    // In this case, just do it in serial
    if (points < total_shepherds)
    {
        // IMPLEMENT 
        cout() << "job is small, running locally" << endl;
        return 0;
    } 

    // {{{ load balancing
    const double points_per_shepherd = double(points) / double(total_shepherds);

    cout() << ( boost::format("%1% point%2% total\n"
                              "%3% point%4% per shepherd\n")
              % points
              % ((points == 1) ? "" : "s")
              % points_per_shepherd
              % ((points_per_shepherd == 1) ? "" : "s")); 

    typedef std::map<gid_type, std::vector<std::size_t> > allocations_type;
    allocations_type allocations;

    double excess = 0;
    for (std::size_t i = 0; i < localities.size(); ++i)
    {
        std::vector<std::size_t>& a = allocations[localities[i]];

        const std::size_t shepherds = topology[localities[i]];
        const double local_points = shepherds * points_per_shepherd;

        cout() << ( boost::format("locality %1%, %2% shepherds, "
                                  "%3% points\n")
                  % get_prefix_from_gid(localities[i])
                  % shepherds
                  % local_points);

        a.insert(a.begin(), shepherds, std::floor(points_per_shepherd));

        const std::size_t local_excess
            = local_points - std::floor(points_per_shepherd) * shepherds;

        cout() << ( boost::format("locality %1% has %2% local excess "
                                  "points\n")
                  % get_prefix_from_gid(localities[i])
                  % local_excess);

        // Round-robin the local excess
        for (std::size_t j = 0; j < local_excess; ++j)
            ++a[j];

        const double global_excess_contribution
            = local_points - std::floor(local_points);

        excess += global_excess_contribution;

        cout() << ( boost::format("locality %1% added %2% to the global "
                                  "excess, global excess is now %3%\n")
                  % get_prefix_from_gid(localities[i])
                  % global_excess_contribution
                  % excess);
    }

    if (!equal(excess, std::floor(excess)))
    {
        cout() << flush;
        cerr() << ( boost::format("error: impossible global excess %1% "
                                  "encountered\n")
                  % excess) << flush; 
        return 1;
    } 

    // Round-robin the global excess
    for (std::size_t i = 0, l = 0; i < std::size_t(excess); ++i, ++l)
        ++allocations[localities[l % localities.size()]]
                     [i % topology[localities[l % localities.size()]]];  

    std::size_t total_points = 0, i = 0, ghost_zones = 0;
    for (allocations_type::const_iterator it = allocations.begin()
                                        , end = allocations.end();
         it != end; ++it, ++i)
    {
        std::size_t local_total = 0;
        for (std::size_t y = 0; y < it->second.size(); ++y) 
        {
            cout() << ( boost::format
                            ("shepherd %1% on locality %2% has %3% points\n")
                      % y
                      % get_prefix_from_gid(it->first)
                      % it->second[y]);
            local_total += it->second[y];
        }

        cout() << ( boost::format("locality %1% has %2% total points\n")
                  % get_prefix_from_gid(it->first)
                  % local_total);

        total_points += local_total;

        // {{{ compute ghost zone sizes
        if ((i + 1) < allocations.size())
        {
            cout() << ( boost::format("ghost zone for locality %1% at %2%\n")
                      % (get_prefix_from_gid(it->first) + 1)
                      % local_total);
            ++ghost_zones;
        } 
        // }}}
    }

    cout() << (boost::format("%1% total points allocated\n") % total_points);
    // }}}

    // {{{ compute total ghost zone size
    cout() << ( boost::format("%1% ghost zones (%2%%% of data duplicated)\n")
              % ghost_zones
              % (double(ghost_zones) / double(total_points)))
           << flush;
    // }}}

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
       desc_commandline("usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ( "points"
        , value<std::size_t>()->default_value(65536)
        , "number of data points") 
        ;

    // Initialize and run HPX
    return init(desc_commandline, argc, argv);
}

