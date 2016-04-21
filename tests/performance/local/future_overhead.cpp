//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// TODO: Update

#include <hpx/hpx_init.hpp>
#include <hpx/lcos/wait_each.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/util/high_resolution_timer.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/iostreams.hpp>

#include <stdexcept>
#include <vector>

#include <boost/format.hpp>
#include <boost/cstdint.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using hpx::init;
using hpx::finalize;

using hpx::find_here;
using hpx::naming::id_type;

using hpx::future;
using hpx::async;
using hpx::lcos::wait_each;

using hpx::util::high_resolution_timer;

using hpx::cout;
using hpx::flush;

///////////////////////////////////////////////////////////////////////////////
// we use globals here to prevent the delay from being optimized away
double global_scratch = 0;
boost::uint64_t num_iterations = 0;

///////////////////////////////////////////////////////////////////////////////
double null_function()
{
    double d = 0.;
    for (boost::uint64_t i = 0; i < num_iterations; ++i)
        d += 1. / (2. * i + 1.);
    return d;
}

HPX_PLAIN_ACTION(null_function, null_action)

struct scratcher
{
    void operator()(future<double> r) const
    {
        global_scratch += r.get();
    }
};

void measure_action_futures(boost::uint64_t count, bool csv)
{
    const id_type here = find_here();

    std::vector<future<double> > futures;
    futures.reserve(count);

    // start the clock
    high_resolution_timer walltime;

    for (boost::uint64_t i = 0; i < count; ++i)
        futures.push_back(async<null_action>(here));

    wait_each(scratcher(), futures);

    // stop the clock
    const double duration = walltime.elapsed();

    if (csv)
        cout << ( boost::format("%1%,%2%\n")
                % count
                % duration)
              << flush;
    else
        cout << ( boost::format("invoked %1% futures (actions) in %2% seconds\n")
                % count
                % duration)
              << flush;
}

void measure_function_futures(boost::uint64_t count, bool csv)
{
    std::vector<future<double> > futures;

    futures.reserve(count);

    // start the clock
    high_resolution_timer walltime;

    for (boost::uint64_t i = 0; i < count; ++i)
        futures.push_back(async(&null_function));

    wait_each(scratcher(), futures);

    // stop the clock
    const double duration = walltime.elapsed();

    if (csv)
        cout << ( boost::format("%1%,%2%\n")
                % count
                % duration)
              << flush;
    else
        cout << ( boost::format("invoked %1% futures (functions) in %2% seconds\n")
                % count
                % duration)
              << flush;
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(
    variables_map& vm
    )
{
    {
        num_iterations = vm["delay-iterations"].as<boost::uint64_t>();

        const boost::uint64_t count = vm["futures"].as<boost::uint64_t>();

        if (HPX_UNLIKELY(0 == count))
            throw std::logic_error("error: count of 0 futures specified\n");

        measure_action_futures(count, vm.count("csv") != 0);
        measure_function_futures(count, vm.count("csv") != 0);
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
        ( "futures"
        , value<boost::uint64_t>()->default_value(500000)
        , "number of futures to invoke")

        ( "delay-iterations"
        , value<boost::uint64_t>()->default_value(0)
        , "number of iterations in the delay loop")

        ( "csv"
        , "output results as csv (format: count,duration)")
        ;

    // Initialize and run HPX.
    return init(cmdline, argc, argv);
}
