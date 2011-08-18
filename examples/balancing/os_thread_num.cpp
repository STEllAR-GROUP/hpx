//  Copyright (c) 2011 Bryce Lelbach 
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/lcos/local_barrier.hpp>

#include <cmath>
#include <vector>
#include <map>

#include <ios>
#include <iomanip>
#include <iostream>

#include <boost/foreach.hpp>
#include <boost/lockfree/fifo.hpp>

using boost::lockfree::fifo;

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using hpx::lcos::local_barrier;

using hpx::threads::threadmanager_base;
using hpx::threads::pending;
using hpx::threads::thread_priority_normal;

using hpx::applier::register_work;

using hpx::init;
using hpx::finalize;

///////////////////////////////////////////////////////////////////////////////
void get_os_thread_num(local_barrier& barr, fifo<std::size_t>& shepherds)
{
    shepherds.enqueue(threadmanager_base::get_thread_num());
    barr.wait();    
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm)
{
    {
        typedef std::map<std::size_t, std::size_t>
            result_map;
    
        typedef std::multimap<
            std::size_t, std::size_t, std::greater<std::size_t>
        > sorter;
    
        std::size_t pxthreads = 0;
    
        if (vm.count("pxthreads"))
            pxthreads = vm["pxthreads"].as<std::size_t>();
        
        std::size_t iterations = 0;
    
        if (vm.count("iterations"))
            iterations = vm["iterations"].as<std::size_t>();
    
        result_map results;
    
        for (std::size_t i = 0; i < iterations; ++i)
        {
            // Have the fifo preallocate the nodes.
            fifo<std::size_t> shepherds(pxthreads); 
    
            local_barrier barr(pxthreads + 1);
   
            const std::size_t one_percent =
                std::floor(double(pxthreads) / 100.0);

            for ( std::size_t percent_count = 0, percent = 0, j = 0
                ; j < pxthreads
                ; ++percent_count, ++j)
            {
                if (percent_count == one_percent && percent < 99)
                {
                    ++percent;
                    percent_count = 0;
                    std::cout << percent << "% of pxthreads created\n";  
                }

                register_work(boost::bind(&get_os_thread_num
                                        , boost::ref(barr)
                                        , boost::ref(shepherds))
                  , "get_os_thread_num"
                  , pending
                  , thread_priority_normal
                  , 0);
            }  

            std::cout << "100% of pxthreads created\n";  
  
            barr.wait(); // wait for all PX threads to enter the barrier
    
            std::size_t shepherd = 0;
    
            while (shepherds.dequeue(&shepherd))
                ++results[shepherd];
        }
    
        sorter sort;
    
        BOOST_FOREACH(result_map::value_type const& result, results)
        { sort.insert(sorter::value_type(result.second, result.first)); }  
        
        BOOST_FOREACH(sorter::value_type const& result, sort) {
            std::cout << std::setfill('0') << std::setw(4)
                      << result.second << " -> "
                      << result.first << "\n";
        }

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
        ("pxthreads,T", value<std::size_t>()->default_value(128), 
            "the number of PX threads to invoke")
        ("iterations", value<std::size_t>()->default_value(1), 
            "the number of times to repeat the test") 
        ;

    // Initialize and run HPX
    return init(desc_commandline, argc, argv);
}

