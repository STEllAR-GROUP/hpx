//  Copyright (c) 2003-2007 Christopher M. Kohlhoff
//  Copyright (c) 2007-2010 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>
#include <string>

#include <boost/cstdint.hpp>
#include <boost/assert.hpp>
#include <boost/lexical_cast.hpp>

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/util/lightweight_test.hpp>
#include <hpx/util/asio_util.hpp>

// no-op
int hpx_main(boost::program_options::variables_map &vm) { return 0; }

int main()
{
    using hpx::naming::locality;
    using hpx::naming::address;
    using hpx::naming::resolver_client;
    using hpx::naming::gid_type;

    using hpx::naming::server::get_command_name;
    using hpx::naming::server::command_lastcommand;

    using hpx::detail::agas_server_helper;

    using hpx::util::get_random_port;
    using hpx::util::io_service_pool;

    boost::uint16_t port = get_random_port();

    std::cout << "Randomized port for AGAS: " << port << std::endl;

    // This is our locality.
    locality here("127.0.0.1", port);

    // Start the AGAS server in the background.
    agas_server_helper agas_server("127.0.0.1", port);

    // Create a client and connect it to the server.
    io_service_pool io_service_pool; 
    resolver_client resolver(io_service_pool, here); 
    
    gid_type last_lowerid;
    
    for (int i = 0; i < 96; ++i)
    {
        // Retrieve the id prefix of this site.
        gid_type prefix1;
        if (resolver.get_prefix(here, prefix1))
            last_lowerid = prefix1;
        HPX_TEST(prefix1 != 0);
    
        std::vector<gid_type> prefixes;
        resolver.get_prefixes(prefixes);
        HPX_TEST((i != 0) ? (2 == prefixes.size()) : (1 == prefixes.size()));
        HPX_TEST_EQ(prefixes.back(), prefix1);
    
         // Identical sites should get same prefix.
        gid_type prefix2;
        HPX_TEST(!resolver.get_prefix(here, prefix2));
        HPX_TEST_EQ(prefix2, prefix1);
    
        // Different sites should get different prefix.
        gid_type prefix3;
        resolver.get_prefix(locality("1.1.1.1", 1), prefix3);
        HPX_TEST(prefix3 != prefix2); 
   
        prefixes.clear(); 
        resolver.get_prefixes(prefixes);
        HPX_TEST_EQ(2U, prefixes.size());
        HPX_TEST_EQ(prefixes.front(), prefix3);
    
        gid_type prefix4;
        HPX_TEST(!resolver.get_prefix(locality("1.1.1.1", 1), prefix4));
        HPX_TEST_EQ(prefix3, prefix4);   
    
        // test get_id_range
        gid_type lower1, upper1;
        HPX_TEST(!resolver.get_id_range(here, 1024, lower1, upper1));
        HPX_TEST(0 == i || last_lowerid + 1 == lower1);   
        
        gid_type lower2, upper2;
        HPX_TEST(!resolver.get_id_range(here, 1024, lower2, upper2));
        HPX_TEST_EQ(upper1 + 1, lower2);   
        last_lowerid = upper2;
        
        // bind an arbitrary address
        HPX_TEST(resolver.bind(gid_type(1), address(here, 1, 2)));
        
        // associate this id with a namespace name
        HPX_TEST(resolver.registerid("/test/foo/1", gid_type(1)));
        
        // resolve this address
        address addr;
        HPX_TEST(resolver.resolve(gid_type(1), addr, false));
        HPX_TEST_EQ(addr, address(here, 1, 2));
    
        // try to resolve a non-existing address
        HPX_TEST(!resolver.resolve(gid_type(2), addr, false));
    
        // check association of the namespace name
        gid_type id;
        HPX_TEST(resolver.queryid("/test/foo/1", id));
        HPX_TEST_EQ(id, gid_type(1));
        
        // rebind the id above to a new address
        HPX_TEST(!resolver.bind(gid_type(1), address(here, 1, 3)));
    
        // re-associate this id with a namespace name
        HPX_TEST(!resolver.registerid("/test/foo/1", gid_type(2)));
    
        // resolve it again
        HPX_TEST(resolver.resolve(gid_type(1), addr, false));
        HPX_TEST_EQ(addr, address(here, 1, 3));
    
        // re-check association of the namespace name
        HPX_TEST(resolver.queryid("/test/foo/1", id));
        HPX_TEST_EQ(id, gid_type(2));
    
        // unbind the address
        HPX_TEST(resolver.unbind(gid_type(1), addr));
        HPX_TEST_EQ(addr, address(here, 1, 3));
    
        // remove association
        HPX_TEST(resolver.unregisterid("/test/foo/1"));
        
        // resolve should fail now
        HPX_TEST(!resolver.resolve(gid_type(1), addr, false));
    
        // association of the namespace name should fail now
        HPX_TEST(!resolver.queryid("/test/foo/1", id));
    
        // repeated unbind should fail
        HPX_TEST(!resolver.unbind(gid_type(1)));
    
        // repeated remove association should fail
        HPX_TEST(!resolver.unregisterid("/test/foo/1"));
        
        // test bind_range/unbind_range API
        HPX_TEST(resolver.bind_range(gid_type(3), 20, address(here, 1, 2), 10));
    
        HPX_TEST(resolver.resolve(gid_type(3), addr, false));
        HPX_TEST_EQ(addr, address(here, 1, 2));
        
        HPX_TEST(resolver.resolve(gid_type(6), addr, false));
        HPX_TEST_EQ(addr, address(here, 1, 32));
    
        HPX_TEST(resolver.resolve(gid_type(22), addr, false));
        HPX_TEST_EQ(addr, address(here, 1, 192));
        
        HPX_TEST(!resolver.resolve(gid_type(23), addr, false));
    
        HPX_TEST(resolver.unbind_range(gid_type(3), 20, addr));
        HPX_TEST_EQ(addr, address(here, 1, 2));
    }

#if 0
    // get statistics
    std::vector<std::size_t> counts;
    std::vector<double> timings;
    std::vector<double> moments;

    HPX_TEST(resolver.get_statistics_count(counts));
    HPX_TEST(resolver.get_statistics_mean(timings));
    HPX_TEST(resolver.get_statistics_moment2(moments));

    std::cout << "Gathered statistics for 96 iterations:\n";

    for (std::size_t i = 0; i < command_lastcommand; ++i)
    {
        std::cout << get_command_name(i) << ": " 
                  << counts[i] << ", " << timings[i] << ", " << moments[i] 
                  << std::endl;
    }
#endif

    return hpx::util::report_errors();
}

