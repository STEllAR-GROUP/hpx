//  Copyright (c) 2011 Bryce Lelbach 
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/integer.hpp>
#include <boost/unordered_set.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/foreach.hpp>
#include <boost/fusion/include/at_c.hpp>

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/util/lightweight_test.hpp>
#include <hpx/util/hash_asio_basic_endpoint.hpp>
#include <hpx/runtime/agas/namespace/user_primary.hpp>
#include <hpx/runtime/agas/database/backend/stdmap.hpp>
#include <hpx/runtime/agas/network/backend/tcpip.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using boost::mt19937;
using boost::uniform_int;

using boost::integer_traits;
 
using boost::asio::ip::address;
using boost::asio::ip::address_v4;
using boost::asio::ip::address_v6;

using boost::fusion::at_c;

using hpx::naming::gid_type;
using hpx::naming::id_type;

using hpx::applier::get_applier;
using hpx::applier::register_work;

using hpx::agas::user_primary_namespace;

using hpx::init;
using hpx::finalize;

using hpx::util::report_errors;

typedef hpx::agas::user_primary_namespace<
    hpx::agas::tag::database::stdmap, hpx::agas::tag::network::tcpip
> primary_namespace_type;

void insert_locality (primary_namespace_type& pri,
                      primary_namespace_type::endpoint_type const& ep,
                      primary_namespace_type::count_type const count)
{
    primary_namespace_type::binding_type const range = pri.bind(ep, count);
    HPX_TEST_EQ(at_c<0>(range).get_msb(), at_c<1>(range).get_msb());
    HPX_TEST_EQ(at_c<1>(range).get_lsb(), count);
}

void resolve_locality (primary_namespace_type& pri,
                       primary_namespace_type::endpoint_type const& ep,
                       primary_namespace_type::count_type const count)
{
    primary_namespace_type::gva_type const gva = at_c<1>(pri.resolve(ep));
    HPX_TEST_EQ(gva.count, count);
}

template <typename Address>
void test_tcpip_primary_namespace(std::size_t entries)
{
    boost::unordered_set<primary_namespace_type::endpoint_type> eps;

    mt19937 rng;

    // Random number distributions.
    uniform_int<typename Address::bytes_type::value_type> byte_dist(1,
        integer_traits<typename Address::bytes_type::value_type>::const_max);
    uniform_int<boost::uint16_t> port_dist(1,
        integer_traits<boost::uint16_t>::const_max);
    uniform_int<primary_namespace_type::count_type> count_dist(1, 0x1000);

    // Get this locality's prefix.
    id_type prefix = get_applier().get_runtime_support_gid();

    // Create the primary namespace.
    primary_namespace_type pri;
    pri.create(prefix);
        
    for (std::size_t e = 0; e < entries; ++e)
    {
        primary_namespace_type::endpoint_type ep;
        typename Address::bytes_type addr;

        // Generate a unique IP address and port
        do {
            typedef typename Address::bytes_type::value_type byte;
            BOOST_FOREACH(byte& b, addr) { b = byte_dist(rng); }
            ep = primary_namespace_type::endpoint_type
                (address(Address(addr)), port_dist(rng)); 
        } while (eps.count(ep));
 
        eps.insert(ep);

        primary_namespace_type::count_type const count = count_dist(rng);

        insert_locality(pri, ep, count); 
        resolve_locality(pri, ep, count);
    }
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm)
{
    std::size_t entries = 0;

    if (vm.count("entries"))
        entries = vm["entries"].as<std::size_t>();
    
    std::size_t iterations = 0;

    if (vm.count("iterations"))
        iterations = vm["iterations"].as<std::size_t>();

    for (std::size_t i = 0; i < iterations; ++i)
    {
        test_tcpip_primary_namespace<address_v4>(entries);
        test_tcpip_primary_namespace<address_v6>(entries);
    }

    // initiate shutdown of the runtime system
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
        ("entries", value<std::size_t>()->default_value(1 << 6), 
            "the number of entries used in each iteration") 
        ("iterations", value<std::size_t>()->default_value(1 << 6), 
            "the number of times to repeat the test") 
        ;

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(init(desc_commandline, argc, argv), 0,
      "HPX main exited with non-zero status");
    return report_errors();
}

