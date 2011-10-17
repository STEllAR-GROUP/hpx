//  Copyright (c) 2011 Bryce Adelstein-Lelbach 
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <stdexcept>

#include <boost/format.hpp>
#include <boost/bind.hpp>
#include <boost/cstdint.hpp>

#include <hpx/hpx_init.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/runtime/threads/threadmanager.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/actions/continuation_impl.hpp>
#include <hpx/runtime/components/plain_component_factory.hpp>
#include <hpx/util/high_resolution_timer.hpp>
#include <hpx/include/iostreams.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using hpx::init;
using hpx::finalize;

using hpx::applier::register_work;
using hpx::applier::get_applier;

using hpx::threads::suspend;
using hpx::threads::threadmanager_base;

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
void null_thread()
{
    global_scratch = delay();
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(
    variables_map& vm
    )
{
    {
        num_iterations = vm["delay-iterations"].as<boost::uint64_t>();
    
        const boost::uint64_t count = vm["pxthreads"].as<boost::uint64_t>();
    
        if (HPX_UNLIKELY(0 == count))
            throw std::logic_error("error: count of 0 pxthreads specified\n");
 
        threadmanager_base& tm = get_applier().get_thread_manager();

        // start the clock 
        high_resolution_timer walltime;
   
        for (boost::uint64_t i = 0; i < count; ++i)
            register_work(boost::bind(&null_thread));

        // Reschedule hpx_main until all other pxthreads have finished. We
        // should be resumed after most of the null pxthreads have been
        // executed. If we haven't, we just reschedule ourselves again.
        do {
            suspend();
        } while (tm.get_thread_count() > 1);

        const double duration = walltime.elapsed();
    
        if (vm.count("csv"))
            cout << ( boost::format("%1%,%2%\n")
                    % count 
                    % duration)
                 << flush;
        else
            cout << ( boost::format("invoked %1% pxthreads in %2% seconds\n")
                    % count
                    % duration)
                 << flush;
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
        ( "pxthreads"
        , value<boost::uint64_t>()->default_value(500000) 
        , "number of pxthreads to invoke")
        
        ( "delay-iterations"
        , value<boost::uint64_t>()->default_value(0) 
        , "number of iterations in the delay loop")

        ( "csv"
        , "output results as csv (format: count,duration)")
        ;

    // Initialize and run HPX.
    return init(cmdline, argc, argv);
}

