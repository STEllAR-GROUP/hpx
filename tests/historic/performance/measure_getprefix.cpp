//  Copyright (c) 2007-2010 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#define MAX_ITERATIONS 100

#include <iostream>
#include <string>
#include <hpx/hpx.hpp>
#include <boost/cstdint.hpp>
#include <boost/assert.hpp>
#include <boost/lexical_cast.hpp>

int main(int argc, char* argv[])
{
    // Check command line arguments.
    std::string host;
    boost::uint16_t port;
    if (argc != 3)
    {
        std::cerr << "Using default settings: localhost:7911" << std::endl;
        std::cerr << "Possible arguments: <AGAS address> <AGAS port>" << std::endl;

        host = "localhost";
        port = 7911;
    }
    else
    {
        host = argv[1];
        port = boost::lexical_cast<boost::uint16_t>(argv[2]);
    }

    try {
        using namespace hpx::naming;

        std::vector<double> timings;
        double total_time = 0;


        // this is our locality
        locality here("localhost", HPX_PORT);
        hpx::util::io_service_pool agas_pool; 
        resolver_client resolver(agas_pool, hpx::naming::locality(host, port));
        
        gid_type last_lowerid;
        
#if defined(MAX_ITERATIONS)
        for (int i = 0; i < MAX_ITERATIONS; ++i)
        {
#endif
        
        // retrieve the id prefix of this site
        gid_type prefix1;
        hpx::util::high_resolution_timer t;
        for(int i=0; i<1000;i++)
        {
            prefix1 = gid_type(i);
            resolver.get_prefix(here, prefix1);
        }
        total_time = total_time + t.elapsed();
#if defined(MAX_ITERATIONS)
        }
        std::cout << "Measure_GetPrefix: "<< total_time / MAX_ITERATIONS << std::endl << std::flush;

        resolver.get_statistics_mean(timings);
        std::cout << " Time taken by get_prefix is: " << timings[0] <<  std::endl <<std::flush;
                
#endif
    }
    catch (std::exception& e) {
        std::cerr << "std::exception caught: " << e.what() << "\n";
        return -1;
    }
    catch (...) {
        std::cerr << "unexpected exception caught\n";
        return -1;
    }
    return 0;
}

