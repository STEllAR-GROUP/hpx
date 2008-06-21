//  Copyright (c) 2007-2008 Hartmut Kaiser
//
//  Parts of this code were taken from the Boost.Asio library
//  Copyright (c) 2003-2007 Christopher M. Kohlhoff (chris at kohlhoff dot com)
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#define MAX_ITERATIONS 100

#include <iostream>
#include <string>

#include <boost/cstdint.hpp>
#include <boost/assert.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/detail/lightweight_test.hpp>

#include <hpx/hpx.hpp>

int main(int argc, char* argv[])
{
    // Check command line arguments.
    std::string host;
    unsigned short port;
    if (argc != 3)
    {
        std::cerr << "Using default settings: localhost:7911" << std::endl;
        std::cerr << "Possible arguments: <DGAS address> <DGAS port>" << std::endl;

        host = "localhost";
        port = 7911;
    }
    else
    {
        host = argv[1];
        port = boost::lexical_cast<unsigned short>(argv[2]);
    }

    try {
        using namespace hpx::naming;

        std::vector<std::size_t> counts;
        std::vector<double> timings;
        std::vector<double> moments;

        // this is our locality
        locality here("localhost", HPX_PORT);
        hpx::util::io_service_pool io_service_pool; 
        resolver_client resolver(io_service_pool, host, port);
        
        id_type last_lowerid;
        
#if defined(MAX_ITERATIONS)
        for (int i = 0; i < MAX_ITERATIONS; ++i)
        {
#endif
        // retrieve the id prefix of this site
        id_type prefix1;
        if (resolver.get_prefix(here, prefix1))
            last_lowerid = prefix1;
        BOOST_TEST(prefix1 != 0);

        std::vector<boost::uint32_t> prefixes;
        resolver.get_prefixes(prefixes);
        BOOST_TEST(((0 == i) ? 1 : 2) == prefixes.size());
        BOOST_TEST(hpx::naming::get_id_from_prefix(prefixes.back()) == prefix1);

        id_type prefix2;
        BOOST_TEST(!resolver.get_prefix(here, prefix2));
        BOOST_TEST(prefix2 == prefix1);   // same site should get same prefix

        // different sites should get different prefix
        id_type prefix3;
        resolver.get_prefix(locality("1.1.1.1", 1), prefix3);
        BOOST_TEST(prefix3 != prefix2);   

        resolver.get_prefixes(prefixes);
        BOOST_TEST(2 == prefixes.size());
        BOOST_TEST(hpx::naming::get_id_from_prefix(prefixes.front()) == prefix3);

        id_type prefix4;
        BOOST_TEST(!resolver.get_prefix(locality("1.1.1.1", 1), prefix4));
        BOOST_TEST(prefix3 == prefix4);   

        // test get_id_range
        id_type lower1, upper1;
        BOOST_TEST(!resolver.get_id_range(here, 1024, lower1, upper1));
        BOOST_TEST(0 == i || last_lowerid+1 == lower1);   
        
        id_type lower2, upper2;
        BOOST_TEST(!resolver.get_id_range(here, 1024, lower2, upper2));
        BOOST_TEST(upper1+1 == lower2);   
        last_lowerid = upper2;
        
        // bind an arbitrary address
        BOOST_TEST(resolver.bind(id_type(1), address(here, 1, 2)));
        
        // associate this id with a namespace name
        BOOST_TEST(resolver.registerid("/test/foo/1", id_type(1)));
        
        // resolve this address
        address addr;
        BOOST_TEST(resolver.resolve(id_type(1), addr));
        BOOST_TEST(addr == address(here, 1, 2));

        // try to resolve a non-existing address
        BOOST_TEST(!resolver.resolve(id_type(2), addr));

        // check association of the namespace name
        id_type id;
        BOOST_TEST(resolver.queryid("/test/foo/1", id));
        BOOST_TEST(id == id_type(1));
        
        // rebind the id above to a new address
        BOOST_TEST(!resolver.bind(id_type(1), address(here, 1, 3)));

        // re-associate this id with a namespace name
        BOOST_TEST(!resolver.registerid("/test/foo/1", id_type(2)));

        // resolve it again
        BOOST_TEST(resolver.resolve(id_type(1), addr));
        BOOST_TEST(addr == address(here, 1, 3));

        // re-check association of the namespace name
        BOOST_TEST(resolver.queryid("/test/foo/1", id));
        BOOST_TEST(id == id_type(2));

        // unbind the address
        BOOST_TEST(resolver.unbind(id_type(1), addr));
        BOOST_TEST(addr == address(here, 1, 3));

        // remove association
        BOOST_TEST(resolver.unregisterid("/test/foo/1"));
        
        // resolve should fail now
        BOOST_TEST(!resolver.resolve(id_type(1), addr));

        // association of the namespace name should fail now
        BOOST_TEST(!resolver.queryid("/test/foo/1", id));

        // repeated unbind should fail
        BOOST_TEST(!resolver.unbind(id_type(1)));

        // repeated remove association should fail
        BOOST_TEST(!resolver.unregisterid("/test/foo/1"));
        
        // test bind_range/unbind_range API
        BOOST_TEST(resolver.bind_range(id_type(3), 20, address(here, 1, 2), 10));

        BOOST_TEST(resolver.resolve(id_type(3), addr));
        BOOST_TEST(addr == address(here, 1, 2));
        
        BOOST_TEST(resolver.resolve(id_type(6), addr));
        BOOST_TEST(addr == address(here, 1, 32));

        BOOST_TEST(resolver.resolve(id_type(22), addr));
        BOOST_TEST(addr == address(here, 1, 192));
        
        BOOST_TEST(!resolver.resolve(id_type(23), addr));

        BOOST_TEST(resolver.unbind_range(id_type(3), 20, addr));
        BOOST_TEST(addr == address(here, 1, 2));

        // get statistics
        BOOST_TEST(resolver.get_statistics_count(counts));
        BOOST_TEST(resolver.get_statistics_mean(timings));
        BOOST_TEST(resolver.get_statistics_moment2(moments));

#if defined(MAX_ITERATIONS)
        }
        
        int iterations = MAX_ITERATIONS;
#else
        int iterations = 1;
#endif

        std::cout << "Gathered statistics for " << iterations 
                  << " iterations: " << std::endl;
        for (std::size_t i = 0; i < server::command_lastcommand; ++i)
        {
            std::cout << server::command_names[i] << ": " 
                      << counts[i] << ", " << timings[i] << ", " << moments[i] 
                      << std::endl;
        }
    }
    catch (std::exception& e) {
        std::cerr << "std::exception caught: " << e.what() << "\n";
    }
    catch (...) {
        std::cerr << "unexpected exception caught\n";
    }
    return boost::report_errors();
}

