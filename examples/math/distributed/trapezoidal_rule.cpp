////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <boost/cstdint.hpp>
#include <boost/format.hpp>
#include <boost/foreach.hpp>

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/components/plain_component_factory.hpp>
#include <hpx/lcos/eager_future.hpp>

#include <examples/math/csv/parse.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using hpx::naming::id_type;
using hpx::naming::gid_type;
using hpx::naming::get_prefix_from_gid;

using hpx::applier::get_applier;

using hpx::actions::plain_result_action0;

using hpx::lcos::eager_future;
using hpx::lcos::future_value;

using hpx::get_runtime;
using hpx::init;
using hpx::finalize;

///////////////////////////////////////////////////////////////////////////////
std::size_t report_shepherd_count();

typedef plain_result_action0<
    // result type
    std::size_t 
    // function
  , report_shepherd_count
> report_shepherd_count_action;

HPX_REGISTER_PLAIN_ACTION(report_shepherd_count_action);

typedef eager_future<report_shepherd_count_action> report_shepherd_count_future;

///////////////////////////////////////////////////////////////////////////////
std::size_t report_shepherd_count()
{ return get_runtime().get_process().get_num_os_threads(); }

///////////////////////////////////////////////////////////////////////////////
double trapezoidal_rule(hpx::math::csv::ast const& points);

// IMPLEMENT

double trapezoidal_rule(hpx::math::csv::ast const& points)
{
    // IMPLEMENT
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int master(variables_map& vm)
{
    // {{{ csv parsing 
    hpx::math::csv::ast points;
    const std::string filename = vm["csv"].as<std::string>();

    if (filename.empty())
    {
        std::cerr << "error: no csv file specified\n";
        return 1;
    }

    switch (hpx::math::csv::parse(filename, points))
    {
        case hpx::math::csv::path_does_not_exist:
            std::cerr << ( boost::format("error: '%1%' does not exist\n")
                         % filename); 
            return 1;
        case hpx::math::csv::path_is_directory:
            std::cerr << ( boost::format("error: '%1%' is a directory\n")
                         % filename); 
            return 1;
        case hpx::math::csv::parse_failed:
            std::cerr << ( boost::format("error: parsing '%1%' failed\n")
                         % filename);
            return 1;
        case hpx::math::csv::parse_succeeded:
            break;
    };

    BOOST_FOREACH(std::vector<double> const& line, points)
    {
        // {{{ debug code to verify parsing
        //for (std::size_t i = 0; i < line.size(); ++i)
        //{
        //    std::cout << line[i] << " ";
        //}
        //std::cout << "\n";
        // }}}

        BOOST_ASSERT(line.size());
    } 
    // }}}

    // {{{ machine discovery 
    std::vector<gid_type> localities;
    get_applier().get_agas_client().get_prefixes(localities);

    std::vector<future_value<std::size_t> > results;
    BOOST_FOREACH(gid_type const& locality, localities)
    { results.push_back(report_shepherd_count_future(locality)); }

    std::size_t total_shepherds = 0;
    std::map<gid_type, std::size_t> topology;

    for (std::size_t i = 0; i < results.size(); ++i)
    {
        const std::size_t shepherds = results[i].get();

        std::cout << ( boost::format("locality %1% has %2% shepherds\n")
                     % get_prefix_from_gid(localities[i])
                     % shepherds);

        topology[localities[i]] = shepherds;
        total_shepherds += shepherds;
    }

    std::cout << ( boost::format("total: %1% localities, %2% shepherds\n")
                 % localities.size()
                 % total_shepherds);
    // }}}

    // In this case, just do it in serial
    if (points.size() < total_shepherds)
    {
        trapezoidal_rule(points);
        return 0;
    } 

    // {{{ load balancing
    const std::size_t points_per_shepherd
        = points.size() / total_shepherds;

    const std::size_t excess
        = points.size() - points_per_shepherd * total_shepherds;

    std::cout << ( boost::format("%1% points total\n"
                                 "%2% point%3% per shepherd, %4% excess\n")
                 % points.size()
                 % points_per_shepherd
                 % ((points_per_shepherd == 1) ? "" : "s")
                 % excess); 

    typedef std::map<gid_type, std::vector<std::size_t> > allocations_type;
    allocations_type allocations;

    for (std::size_t i = 0; i < localities.size(); ++i)
    {
        std::vector<std::size_t>& a = allocations[localities[i]];
        a.insert(a.begin(), topology[localities[i]], points_per_shepherd);
    }

    // Round-robin the reminder
    for (std::size_t i = 0, l = 0; i < excess; ++i, ++l)
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
            std::cout << ( boost::format
                            ("shepherd %1% on locality %2% has %3% points\n")
                         % y
                         % get_prefix_from_gid(it->first)
                         % it->second[y]);
            local_total += it->second[y];
        }

        std::cout << ( boost::format("locality %1% has %2% total points\n")
                     % get_prefix_from_gid(it->first)
                     % local_total);

        total_points += local_total;

        // {{{ compute ghost zone sizes
        if ((i + 1) < allocations.size())
        {
            std::cout << ( boost::format("ghost zone for locality %1% at %2%\n")
                         % (get_prefix_from_gid(it->first) + 1)
                         % local_total);
            ++ghost_zones;
        } 
        // }}}
    }

    std::cout << (boost::format("%1% total points allocated\n") % total_points);

    // {{{ compute total ghost zone size
    std::cout << ( boost::format("%1% ghost zones (%2%%% of data duplicated)\n")
                 % ghost_zones
                 % (double(ghost_zones) / double(total_points)));
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
        ( "csv"
        , value<std::string>()->default_value("")
        , "csv file containing the dataset (2D coordinates)") 
        ;

    // Initialize and run HPX
    return init(desc_commandline, argc, argv);
}

