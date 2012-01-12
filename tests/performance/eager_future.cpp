//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/lcos/async_future_wait.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/runtime/components/plain_component_factory.hpp>
#include <hpx/util/high_resolution_timer.hpp>
#include <hpx/lcos/eager_future.hpp>
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

using hpx::find_here;

using hpx::naming::id_type;

using hpx::actions::plain_result_action0;

using hpx::lcos::promise;
using hpx::lcos::eager_future;
using hpx::lcos::wait;

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
        d += 1 / (2. * i + 1);
    return d;
}

typedef plain_result_action0<
    // result type
    double
    // arguments
    // function
  , null_function
> null_action;

HPX_REGISTER_PLAIN_ACTION(null_action);

typedef eager_future<null_action> null_future;

///////////////////////////////////////////////////////////////////////////////
int hpx_main(
    variables_map& vm
    )
{
    {
        num_iterations = vm["delay-iterations"].as<boost::uint64_t>();

        const boost::uint64_t count = vm["futures"].as<boost::uint64_t>();

        const id_type here = find_here();

        if (HPX_UNLIKELY(0 == count))
            throw std::logic_error("error: count of 0 futures specified\n");

        std::vector<promise<double> > futures;

        futures.reserve(count);

        // start the clock
        high_resolution_timer walltime;

        for (boost::uint64_t i = 0; i < count; ++i)
            futures.push_back(null_future(here));

        wait(futures, [&] (std::size_t, double r) { global_scratch += r; });

        // stop the clock
        const double duration = walltime.elapsed();

        if (vm.count("csv"))
            cout << ( boost::format("%1%,%2%\n")
                    % count
                    % duration)
                 << flush;
        else
            cout << ( boost::format("invoked %1% futures in %2% seconds\n")
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

