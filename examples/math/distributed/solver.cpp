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
#include <boost/rational.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/split_free.hpp>

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/lcos/eager_future.hpp>
#include <hpx/runtime/components/component_factory.hpp>
#include <hpx/runtime/components/plain_component_factory.hpp>
#include <hpx/runtime/components/stubs/runtime_support.hpp>
#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

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
using hpx::components::stubs::runtime_support::create_component;

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

namespace boost { namespace serialization
{

template <typename Archive, typename T>
void save(Archive& ar, boost::rational<T> const& r, const unsigned int)
{
    T num = r.numerator(), den = r.denominator(); 
    ar & num;
    ar & den;
}

template <typename Archive, typename T>
void load(Archive& ar, boost::rational<T>& r, const unsigned int)
{
    T num(0), den(0):
    ar & num;
    ar & den;
    r.assign(num, den);
}

template <typename Archive, typename T>
void serialize(Archive& ar, boost::rational<T>& r, const unsigned int version)
{ split_free(ar, r, version); }

}}

///////////////////////////////////////////////////////////////////////////////
typedef boost::rational<boost::uint64_t> rational;

HPX_REGISTER_COMPONENT_MODULE();

HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(
    hpx::lcos::base_lco_with_value<rational>,
    component_base_lco_with_value);

HPX_REGISTER_ACTION_EX(
    hpx::lcos::base_lco_with_value<
        rational
    >::set_result_action,
    base_lco_with_value_set_result_rational);

///////////////////////////////////////////////////////////////////////////////
template <typename Iterator, typename Container>
inline Iterator& circular_next(Iterator& it, Container const& c)
{
    if (HPX_UNLIKELY(c.end() == it))
        it = c.begin();
    else
        ++it;
    return it; 
}

///////////////////////////////////////////////////////////////////////////////
rational math_function (rational const& r)
{
    // IMPLEMENT
    return rational(0); 
}

///////////////////////////////////////////////////////////////////////////////
typedef hpx::balancing::server::integrator<math_function, rational> 
    integrator_type;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    hpx::components::managed_component<
        integrator_type
    >, 
    integrator_math_function_factory);

HPX_REGISTER_ACTION_EX(
    integrator_type::build_network_action,
    integrator_math_function_build_network_action);

HPX_REGISTER_ACTION_EX(
    integrator_type::deploy_action,
    integrator_math_function_deploy_action);

HPX_REGISTER_ACTION_EX(
    integrator_type::solve_action,
    integrator_math_function_solve_action);

HPX_REGISTER_ACTION_EX(
    integrator_type::regrid_action,
    integrator_math_function_regrid_action);

HPX_DEFINE_GET_COMPONENT_TYPE(integrator_type);

///////////////////////////////////////////////////////////////////////////////
int master(variables_map& vm)
{
    cout() << std::setprecision(12);
    cerr() << std::setprecision(12);

    // Get options.
    const rational lower_bound = vm["lower-bound"].as<rational>();
    const rational upper_bound = vm["upper-bound"].as<rational>();
    const rational tolerance = vm["tolerance"].as<rational>();
    const rational top_segs = vm["top-segments"].as<rational>();
    const rational regrid_segs = vm["regrid-segments"].as<rational>();

    // Handle for the root discovery component. 
    discovery root;

    // Create the first discovery component on this locality.
    root.create(find_here());

    cout() << "deploying discovery infrastructure\n";

    // Deploy the scheduling infrastructure.
    std::vector<id_type> network = root.build_network_sync(); 

    // Get this localities topology map LVA.
    topology_map const* topology_ptr
        = reinterpret_cast<topology_map const*>(root.topology_lva_sync());

    topology_map const& topology = *topology_ptr;

    boost::uint32_t total_shepherds = 0;
    BOOST_FOREACH(topology_map::value_type const& v, topology)
    {
        cout() << ( boost::format("locality %1% has %2% shepherds\n")
                  % v.first
                  % v.second);
        total_shepherds += v.second;
    }
    
    cout() << ( boost::format("%1% localities, %2% shepherds total\n")
              % topology.size()
              % total_shepherds);

    integrator<math_function, rational> client;

    client.create(find_here());

    client.build_network_sync(network, tolerance, regrid_segs); 

    high_resolution_timer t;

    rational r = client.solve_sync(lower_bound, upper_bound, top_segs);

    double elapsed = t.elapsed();
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
        ( "lower-bound"
        , value<rational>()->default_value(rational(0), "0") 
        , "lower bound of integration")

        ( "upper-bound"
        , value<rational>()->default_value(rational(2815, 7), "64*pi")
        , "upper bound of integration")

        ( "tolerance"
        , value<rational>()->default_value(rational(0, 10), "0.1") 
        , "resolution tolerance")

        ( "top-segments"
        , value<rational>()->default_value(rational(4096), "4096") 
        , "number of top-level segments")

        ( "regrid-segments"
        , value<rational>()->default_value(rational(128), "128") 
        , "number of segment per regrid")
        ;

    // Initialize and run HPX
    return init(desc_commandline, argc, argv);
}

