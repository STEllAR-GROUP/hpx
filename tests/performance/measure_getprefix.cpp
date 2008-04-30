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
#include <hpx/hpx.hpp>
#include <boost/cstdint.hpp>
#include <boost/assert.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/detail/lightweight_test.hpp>
#include <hpx/util/high_resolution_timer.hpp>


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

        std::vector<double> timings;

        // this is our locality
        locality here("localhost", HPX_PORT);
        resolver_client resolver(host, port);
        
        id_type last_lowerid;
        
#if defined(MAX_ITERATIONS)
        for (int i = 0; i < MAX_ITERATIONS; ++i)
        {
#endif
        
	// retrieve the id prefix of this site
	id_type prefix1;		
	hpx::util::high_resolution_timer t;		
	for(int i=0; i<1000;i++)
		{
		 prefix1 = id_type(i);
        	 resolver.get_prefix(here, prefix1);
	        }
	std::cout << " ***************************************"<< std::endl << std::flush;
	std::cout << "Measure_GetPrefix: "<< t.elapsed() << std::endl << std::flush;
		
	resolver.get_statistics(timings);
	std::cout << " Time taken by get_prefix is: " << timings[0] <<  std::endl <<std::flush;
		        
	return 0;
        
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
                      << timings[i] << std::endl;
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

