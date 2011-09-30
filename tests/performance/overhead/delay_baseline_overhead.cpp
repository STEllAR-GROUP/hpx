//  Copyright (c) 2011 Bryce Adelstein-Lelbach 
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <stdexcept>

#include <boost/format.hpp>
#include <boost/cstdint.hpp>

#include <hpx/hpx_init.hpp>
#include <hpx/util/high_resolution_timer.hpp>
#include <hpx/include/iostreams.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using hpx::init;
using hpx::finalize;

using hpx::util::high_resolution_timer;

using hpx::cout;
using hpx::flush;

///////////////////////////////////////////////////////////////////////////////
// we use globals here to prevent the delay from being optimized away
double global_scratch = 0;
boost::uint64_t num_iterations = 0;

///////////////////////////////////////////////////////////////////////////////
double delay()
{
    double d = 0.;
    for (boost::uint64_t i = 0; i < num_iterations; ++i)
        d += 1 / (2. * i + 1);
    return d;
}
    
///////////////////////////////////////////////////////////////////////////////
int hpx_main(
    variables_map& vm
    )
{
    {
        num_iterations = vm["delay-iterations"].as<boost::uint64_t>();
    
        const boost::uint64_t count = vm["count"].as<boost::uint64_t>();

        const bool csv = vm.count("csv");
    
        if (HPX_UNLIKELY(0 == count))
            throw std::logic_error("error: count of 0 loops specified\n");

        for (boost::uint64_t i = 0; i < count; ++i)
        {
            // start the clock 
            high_resolution_timer walltime;
     
            global_scratch = delay();  
      
            // stop the clock 
            const double duration = walltime.elapsed();
        
            if (csv)
                cout << ( boost::format("%1%,%2%\n")
                        % num_iterations 
                        % duration)
                     << flush;
            else
                cout << ( boost::format("ran %1% iterations in %2% seconds\n")
                        % num_iterations
                        % duration)
                     << flush;
        }
    }

    finalize();
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(
    int argc
  , char* argv[]
    )
{
    // Configure application-specific options.
    options_description cmdline("usage: " HPX_APPLICATION_STRING " [options]");

    cmdline.add_options()
        ( "count"
        , value<boost::uint64_t>()->default_value(64) 
        , "number of delay loops to run")
        
        ( "delay-iterations"
        , value<boost::uint64_t>()->default_value(65536) 
        , "number of iterations in the delay loop")

        ( "csv"
        , "output results as csv (format: iterations,duration)")
        ;

    // Initialize and run HPX.
    return init(cmdline, argc, argv);
}

