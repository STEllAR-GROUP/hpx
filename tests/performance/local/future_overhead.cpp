//  Copyright (c) 2018 Mikael Simberg
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// TODO: Update

#include <hpx/hpx_init.hpp>
#include <hpx/lcos/wait_each.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/util/format.hpp>
#include <hpx/util/high_resolution_timer.hpp>
#include <hpx/include/apply.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/util/yield_while.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using hpx::init;
using hpx::finalize;

using hpx::find_here;
using hpx::naming::id_type;

using hpx::future;
using hpx::async;
using hpx::apply;
using hpx::lcos::wait_each;

using hpx::util::high_resolution_timer;

using hpx::cout;
using hpx::flush;

///////////////////////////////////////////////////////////////////////////////
// we use globals here to prevent the delay from being optimized away
double global_scratch = 0;
std::uint64_t num_iterations = 0;

///////////////////////////////////////////////////////////////////////////////
double null_function()
{
    double d = 0.;
    for (std::uint64_t i = 0; i < num_iterations; ++i)
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

void measure_action_futures(std::uint64_t count, bool csv)
{
    const id_type here = find_here();

    std::vector<future<double> > futures;
    futures.reserve(count);

    // start the clock
    high_resolution_timer walltime;

    for (std::uint64_t i = 0; i < count; ++i)
        futures.push_back(async<null_action>(here));

    wait_each(scratcher(), futures);

    // stop the clock
    const double duration = walltime.elapsed();

    if (csv)
        hpx::util::format_to(cout,
            "{1},{2}\n",
            count,
            duration) << flush;
    else
        hpx::util::format_to(cout,
            "invoked {1} futures (actions) in {2} seconds\n",
            count,
            duration) << flush;
    // CDash graph plotting
    hpx::util::print_cdash_timing("FutureOverheadActions", duration);
}

void measure_function_futures_wait_each(std::uint64_t count, bool csv)
{
    std::vector<future<double> > futures;

    futures.reserve(count);

    // start the clock
    high_resolution_timer walltime;

    for (std::uint64_t i = 0; i < count; ++i)
        futures.push_back(async(&null_function));

    wait_each(scratcher(), futures);

    // stop the clock
    const double duration = walltime.elapsed();

    if (csv)
        hpx::util::format_to(cout,
            "{1},{2}\n",
            count,
            duration) << flush;
    else
        hpx::util::format_to(cout,
            "invoked {1} futures (functions, wait_each) in {2} seconds\n",
            count,
            duration) << flush;
    // CDash graph plotting
    hpx::util::print_cdash_timing("FutureOverheadWaitEach", duration);
}

void measure_function_futures_wait_all(std::uint64_t count, bool csv)
{
    std::vector<future<double> > futures;

    futures.reserve(count);

    // start the clock
    high_resolution_timer walltime;

    for (std::uint64_t i = 0; i < count; ++i)
        futures.push_back(async(&null_function));

    wait_all(futures);

    // stop the clock
    const double duration = walltime.elapsed();

    if (csv)
        hpx::util::format_to(cout,
            "{1},{2}\n",
           count,
           duration) << flush;
    else
        hpx::util::format_to(cout,
            "invoked {1} futures (functions, wait_all) in {2} seconds\n",
            count,
            duration) << flush;
    // CDash graph plotting
    hpx::util::print_cdash_timing("FutureOverheadFuturesWait", duration);
}

void measure_function_futures_thread_count(std::uint64_t count, bool csv)
{
    std::vector<future<double> > futures;

    futures.reserve(count);

    // start the clock
    high_resolution_timer walltime;

    for (std::uint64_t i = 0; i < count; ++i)
        apply(&null_function);

    // Yield until there is only this and background threads left.
    auto this_pool = hpx::this_thread::get_pool();
    hpx::util::yield_while([this_pool]()
        {
            return this_pool->get_thread_count_unknown(std::size_t(-1), false) >
                this_pool->get_background_thread_count() + 1;
        });

    // stop the clock
    const double duration = walltime.elapsed();

    if (csv)
        hpx::util::format_to(cout,
            "{1},{2}\n",
            count,
            duration) << flush;
    else
        hpx::util::format_to(cout,
            "invoked {1} futures (functions, thread count) in {2} seconds\n",
            count,
            duration) << flush;
    // CDash graph plotting
    hpx::util::print_cdash_timing("FutureOverheadThreadCount", duration);
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(
    variables_map& vm
    )
{
    {
        num_iterations = vm["delay-iterations"].as<std::uint64_t>();

        const std::uint64_t count = vm["futures"].as<std::uint64_t>();

        if (HPX_UNLIKELY(0 == count))
            throw std::logic_error("error: count of 0 futures specified\n");

        measure_action_futures(count, vm.count("csv") != 0);
        measure_function_futures_wait_each(count, vm.count("csv") != 0);
        measure_function_futures_wait_all(count, vm.count("csv") != 0);
        measure_function_futures_thread_count(count, vm.count("csv") != 0);
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
        , value<std::uint64_t>()->default_value(500000)
        , "number of futures to invoke")

        ( "delay-iterations"
        , value<std::uint64_t>()->default_value(0)
        , "number of iterations in the delay loop")

        ( "csv"
        , "output results as csv (format: count,duration)")
        ;

    // Initialize and run HPX.
    return init(cmdline, argc, argv);
}
