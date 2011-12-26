//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/util.hpp>
#include <hpx/include/iostreams.hpp>

#include <stdexcept>

#include <boost/format.hpp>
#include <boost/bind.hpp>
#include <boost/cstdint.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using hpx::init;
using hpx::finalize;
using hpx::get_os_thread_count;

using hpx::applier::register_work;

using hpx::threads::suspend;
using hpx::threads::get_thread_count;

using hpx::util::high_resolution_timer;

using hpx::cout;
using hpx::flush;

///////////////////////////////////////////////////////////////////////////////
// we use globals here to prevent the delay from being optimized away
double global_scratch = 0;
double global_delay = 0;
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
    if (num_iterations == 0)
        return;

    high_resolution_timer walltime;
    global_scratch = delay();
    global_delay = walltime.elapsed();
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(
    variables_map& vm
    )
{
    {
        num_iterations = vm["delay-iterations"].as<boost::uint64_t>();

        boost::uint64_t const count = vm["px-threads"].as<boost::uint64_t>();
        if (HPX_UNLIKELY(0 == count))
            throw std::logic_error("error: count of 0 px-threads specified\n");

        // start the clock
        high_resolution_timer walltime;

        for (boost::uint64_t i = 0; i < count; ++i)
            register_work(HPX_STD_BIND(&null_thread));

        // Reschedule hpx_main until all other px-threads have finished. We
        // should be resumed after most of the null px-threads have been
        // executed. If we haven't, we just reschedule ourselves again.
        do {
            suspend();
        } while (get_thread_count() > 1);

        double const duration = walltime.elapsed();

        if (vm.count("csv")) {
            if (0 != num_iterations) {
                cout << ( boost::format("%1%,%2%,%3%,%4%,%5%\n")
                        % get_os_thread_count()
                        % global_delay
                        % count
                        % duration
                        % (duration / ((count * global_delay) / get_os_thread_count())))
                     << flush;
            }
            else {
                global_delay = 0;
                cout << ( boost::format("%1%,%2%,%3%,%4%\n")
                        % get_os_thread_count()
                        % global_delay
                        % count
                        % duration)
                     << flush;
            }
        }
        else {
            cout << ( boost::format("invoked %1% px-threads in %2% seconds\n")
                    % count
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
        ( "px-threads"
        , value<boost::uint64_t>()->default_value(500000)
        , "number of px-threads to invoke")

        ( "delay-iterations"
        , value<boost::uint64_t>()->default_value(0)
        , "number of iterations in the delay loop")

        ( "csv"
        , "output results as csv (format: count,duration)")
        ;

    // Initialize and run HPX.
    return init(cmdline, argc, argv);
}

