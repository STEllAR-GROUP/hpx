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
#include <hpx/util/annotated_function.hpp>

#include <cstddef>
//
#include <hpx/include/parallel_execution.hpp>
#include <hpx/runtime/threads/executors/pool_executor.hpp>
#include <hpx/runtime/threads/executors/limiting_executor.hpp>
//
#include <hpx/lcos/local/sliding_semaphore.hpp>
//
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
void print_stats(const char *title, const char *wait, const char *exec, std::int64_t count, double duration, bool csv)
{
    double us = 1e6*duration/count;
    if (csv)
        hpx::util::format_to(cout,
            "{1},{2},{3},{4},{5},{6},\n",
           count, title, duration, us) << flush;
    else
        hpx::util::format_to(cout,
            "invoked {1}, futures {:10} {:15} {:20} in \t{5} seconds \t: {6} us per future \n",
            count, title, wait, exec, duration, us) << flush;
    // CDash graph plotting
    //hpx::util::print_cdash_timing(title, duration);
}

const char *ExecName(const hpx::parallel::execution::parallel_executor &exec) {
    return "parallel_executor";
}
const char *ExecName(const hpx::parallel::execution::default_executor &exec) {
    return "default_executor";
}

///////////////////////////////////////////////////////////////////////////////
// we use globals here to prevent the delay from being optimized away
double global_scratch = 0;
std::uint64_t num_iterations = 0;

///////////////////////////////////////////////////////////////////////////////
double null_function()
{
    if (num_iterations>0) {
        const int array_size = 4096;
        std::array<double, array_size> dummy;
        for (std::uint64_t i = 0; i < num_iterations; ++i) {
            for (std::uint64_t j = 0; j < array_size; ++j) {
                dummy[j] = 1.0 / (2.0 * i * j + 1.0);
            }
        }
        return dummy[0];
    }
    return 0.0;
}

HPX_PLAIN_ACTION(null_function, null_action)

struct scratcher
{
    void operator()(future<double> r) const
    {
        global_scratch += r.get();
    }
};

// Time async action execution using wait each on futures vector
void measure_action_futures_wait_each(std::uint64_t count, bool csv)
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
    print_stats("Actions", "WaitEach", "no-executor", count, duration, csv);
}

// Time async action execution using wait each on futures vector
void measure_action_futures_wait_all(std::uint64_t count, bool csv)
{
    const id_type here = find_here();
    std::vector<future<double> > futures;
    futures.reserve(count);

    // start the clock
    high_resolution_timer walltime;
    for (std::uint64_t i = 0; i < count; ++i)
        futures.push_back(async<null_action>(here));
    wait_all(futures);

    // stop the clock
    const double duration = walltime.elapsed();
    print_stats("Actions", "WaitAll", "no-executor", count, duration, csv);
}

// Time async execution using wait each on futures vector
template <typename Executor>
void measure_function_futures_wait_each(std::uint64_t count, bool csv, Executor &exec)
{
    std::vector<future<double> > futures;
    futures.reserve(count);

    // start the clock
    high_resolution_timer walltime;
    for (std::uint64_t i = 0; i < count; ++i)
        futures.push_back(async(exec, &null_function));
    wait_each(scratcher(), futures);

    // stop the clock
    const double duration = walltime.elapsed();
    print_stats("async", "WaitEach", ExecName(exec), count, duration, csv);
}

template <typename Executor>
void measure_function_futures_wait_all(std::uint64_t count, bool csv, Executor &exec)
{
    std::vector<future<double> > futures;
    futures.reserve(count);

    // start the clock
    high_resolution_timer walltime;
    for (std::uint64_t i = 0; i < count; ++i)
        futures.push_back(async(exec, &null_function));
    wait_all(futures);

    const double duration = walltime.elapsed();
    print_stats("async", "WaitAll", ExecName(exec), count, duration, csv);
}

template <typename Executor>
void measure_function_futures_thread_count(std::uint64_t count, bool csv, Executor &exec)
{
    std::vector<future<double> > futures;
    futures.reserve(count);

    std::atomic<std::uint64_t> sanity_check(count);
    auto this_pool = hpx::this_thread::get_pool();
    // start the clock
    high_resolution_timer walltime;
    for (std::uint64_t i = 0; i < count; ++i) {
        hpx::apply(exec,
            [&sanity_check]() {
               null_function();
               sanity_check--;
            }
        );
    }

    // Yield until there is only this and background threads left.
    hpx::util::yield_while([this_pool, &sanity_check]()
        {
            return (sanity_check>0);
//        auto u = this_pool->get_thread_count_unknown(std::size_t(-1), false);
//        auto b = this_pool->get_background_thread_count() + 1;
////        std::cout << "Unknown " << u << " background " << b << std::endl;
//        return u>b;
//            return this_pool->get_thread_count_unknown(std::size_t(-1), false) >
//                this_pool->get_background_thread_count() + 1;
        });

    // stop the clock
    const double duration = walltime.elapsed();

    if (sanity_check!=0) {
        int count = this_pool->get_thread_count_unknown(std::size_t(-1), false);
        throw std::runtime_error("This test is faulty " + std::to_string(count));
    }

    print_stats("apply", "ThreadCount", ExecName(exec), count, duration, csv);
}

template <typename Executor>
void measure_function_futures_limiting_executor(std::uint64_t count, bool csv, Executor exec)
{
    using namespace hpx::parallel::execution;
    std::uint64_t const num_threads = hpx::get_num_worker_threads();
    std::uint64_t const tasks = num_threads*2000;
    std::atomic<std::uint64_t> sanity_check(count);

    // start the clock
    high_resolution_timer walltime;
    {
        hpx::threads::executors::limiting_executor<Executor> signal_exec(exec, tasks, tasks+1000);
        for (std::uint64_t i = 0; i < count; ++i) {
            hpx::apply(signal_exec, [&](){
                null_function();
                sanity_check--;
            });
        }
    }

    if (sanity_check!=0) {
        throw std::runtime_error("This test is faulty " + std::to_string(sanity_check));
    }

    // stop the clock
    const double duration = walltime.elapsed();
    print_stats("apply", "limiting-Exec", ExecName(exec), count, duration, csv);
}

template <typename Executor>
void measure_function_futures_sliding_semaphore(std::uint64_t count, bool csv, Executor &exec)
{
    std::vector<future<double> > futures;
    futures.reserve(count);

    // start the clock
    high_resolution_timer walltime;
    const int sem_count = 5000;
    hpx::lcos::local::sliding_semaphore sem(sem_count);
    for (std::uint64_t i = 0; i < count; ++i) {
        hpx::async(exec, [i,&sem](){
            null_function();
            sem.signal(i);
        });
        sem.wait(i);
    }
    sem.wait(count);

    // stop the clock
    const double duration = walltime.elapsed();
    print_stats("Apply", "Sliding-Sem", ExecName(exec), count, duration, csv);
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm)
{
    {
        num_iterations = vm["delay-iterations"].as<std::uint64_t>();

        const std::uint64_t count = vm["futures"].as<std::uint64_t>();
        bool csv = vm.count("csv") != 0;
        if (HPX_UNLIKELY(0 == count))
            throw std::logic_error("error: count of 0 futures specified\n");
        const int nl = 1;

        hpx::parallel::execution::default_executor def;
        hpx::parallel::execution::parallel_executor par;

        for (int i=0; i<nl; i++) {
            measure_action_futures_wait_each(count, csv);
            measure_action_futures_wait_all(count, csv);
            measure_function_futures_wait_each(count, csv, def);
            measure_function_futures_wait_each(count, csv, par);
            measure_function_futures_wait_all(count, csv, def);
            measure_function_futures_wait_all(count, csv, par);
            measure_function_futures_thread_count(count, csv, def);
            measure_function_futures_thread_count(count, csv, par);
            measure_function_futures_limiting_executor(count, csv, def);
            measure_function_futures_limiting_executor(count, csv, par);
            measure_function_futures_sliding_semaphore(count, csv, def);
        }
    }

    return hpx::finalize();
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
