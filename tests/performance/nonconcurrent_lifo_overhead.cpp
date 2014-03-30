////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2014 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

// Makes HPX use BOOST_ASSERT, so that I can use high_resolution_timer without
// depending on the rest of HPX.
#define HPX_USE_BOOST_ASSERT

#include "worker_timed.hpp"

#include <stdexcept>
#include <iostream>

#include <boost/thread/thread.hpp>
#include <boost/thread/barrier.hpp>
#include <boost/cstdint.hpp>
#include <boost/format.hpp>
#include <boost/lockfree/stack.hpp>
#include <boost/date_time/gregorian/gregorian.hpp>
#include <boost/program_options.hpp>

#include <hpx/util/high_resolution_timer.hpp>

char const* benchmark_name = "Serial LIFO Overhead";

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;
using boost::program_options::store;
using boost::program_options::command_line_parser;
using boost::program_options::notify;

using hpx::util::high_resolution_timer;

///////////////////////////////////////////////////////////////////////////////
boost::uint64_t threads = 1;
//boost::uint64_t delay = 5;
boost::uint64_t blocksize = 10000;
boost::uint64_t iterations = 2000000;
bool header = true;

///////////////////////////////////////////////////////////////////////////////
std::string format_build_date(std::string timestamp)
{
    boost::gregorian::date d = boost::gregorian::from_us_string(timestamp);

    char const* fmt = "%02i-%02i-%04i";

    return boost::str(boost::format(fmt)
                     % d.month().as_number() % d.day() % d.year());
}

///////////////////////////////////////////////////////////////////////////////
void print_results(
    variables_map& vm
//  , std::pair<double, double> elapsed_stl
  , std::pair<double, double> elapsed_lockfree 
    )
{
    if (header)
    {
        std::cout << "# BENCHMARK: " << benchmark_name << "\n";

        std::cout << "# VERSION: " << format_build_date(__DATE__) << "\n"
             << "#\n";

        // Note that if we change the number of fields above, we have to
        // change the constant that we add when printing out the field # for
        // performance counters below (e.g. the last_index part).
        std::cout <<
//            "## 0:DELAY:Delay [micro-seconds] - Independent Variable\n"
            "## 0:ITER:Iterations per OS-thread - Independent Variable\n"
            "## 1:OSTHRDS:OS-thread - Independent Variable\n"
//            "## 3:WTIME_STL_PUSH:Total Walltime/Push for "
//                "std::deque [nanoseconds]\n"
//            "## 4:WTIME_STL_POP:Total Walltime/Pop for "
//                "std::deque [nanoseconds]\n"
            "## 2:WTIME_LF_PUSH:Total Walltime/Push for "
                "boost::lockfree::stack [nanoseconds]\n"
            "## 3:WTIME_LF_POP:Total Walltime/Pop for "
                "boost::lockfree::stack [nanoseconds]\n"
                ;
    }

    if (iterations != 0)
        std::cout << ( boost::format("%lu %lu %.14g %.14g\n")
//                % delay
                % iterations
                % threads
                % ((elapsed_lockfree.first / (threads*iterations)) * 1e9)
                % ((elapsed_lockfree.second / (threads*iterations)) * 1e9)
                );
    else
        std::cout << ( boost::format("%lu %lu %.14g %.14g\n")
//                % delay
                % iterations
                % threads
                % (elapsed_lockfree.first * 1e9)
                % (elapsed_lockfree.second * 1e9)
                );
}

///////////////////////////////////////////////////////////////////////////////
/*
template <typename T>
struct control_case
{
    typedef T value_type;
};

template <typename T>
void push(control_case<T>& lifo, T& seed)
{
}

template <typename T>
void push(std::deque<T>& lifo, T& seed)
{
    seed ^= 0xAAAA;
    lifo.push_front(seed);
}
*/

template <typename T>
void push(boost::lockfree::stack<T>& lifo, T& seed)
{
//    seed ^= 0xAAAA;
    lifo.push(seed);
}

/*
template <typename T>
void pop(control_case<T>& lifo)
{
}

template <typename T>
void pop(std::deque<T>& lifo)
{
    lifo.pop_back();
}
*/

template <typename T>
void pop(boost::lockfree::stack<T>& lifo)
{
    T t;
    lifo.pop(t);
}

template <typename Lifo>
std::pair<double, double>
bench_lifo(Lifo& lifo, boost::uint64_t local_iterations)
{
    ///////////////////////////////////////////////////////////////////////////
    // Push.
    typename Lifo::value_type seed;

    std::pair<double, double> elapsed(0.0, 0.0);

    // Start the clock.
    high_resolution_timer t;

    for ( boost::uint64_t block = 0
        ; block < (local_iterations / blocksize)
        ; ++block)
    { 
        // Restart the clock.
        t.restart();

        for (boost::uint64_t i = 0; i < blocksize; ++i)
        {
            push(lifo, seed);
        }
    
        elapsed.first += t.elapsed();
    
        ///////////////////////////////////////////////////////////////////////////
        // Pop.
    
        // Restart the clock.
        t.restart(); 
    
        for (boost::uint64_t i = 0; i < blocksize; ++i)
        {
            pop(lifo);
        }
    
        elapsed.second += t.elapsed();
    }
         
    return elapsed; 
}

///////////////////////////////////////////////////////////////////////////////
void perform_iterations(
    boost::barrier& b
//  , std::pair<double, double>& elapsed_stl
  , std::pair<double, double>& elapsed_lockfree
    )
{
    {
//        std::deque<boost::uint64_t> lifo;
//        lifo.reserve(iterations);
//        control_case<boost::uint64_t> control;
//        elapsed_stl = bench_lifo(control);
    }

    {
        boost::lockfree::stack<boost::uint64_t> lifo(1);
        lifo.reserve(iterations);

        // Warmup.
        bench_lifo(lifo, blocksize);

        elapsed_lockfree = bench_lifo(lifo, iterations);
    }
}

///////////////////////////////////////////////////////////////////////////////
int app_main(
    variables_map& vm
    )
{
//    if (0 == iterations)
//        throw std::invalid_argument("error: count of 0 iterations specified\n");

    std::vector<std::pair<double, double> >
        elapsed_stl(threads, std::pair<double, double>(0.0, 0.0));
    std::vector<std::pair<double, double> >
        elapsed_lockfree(threads, std::pair<double, double>(0.0, 0.0));
    boost::thread_group workers;
    boost::barrier b(threads);

    for (boost::uint32_t i = 0; i != threads; ++i)
        workers.add_thread(new boost::thread(
            perform_iterations,
            boost::ref(b),
//            boost::ref(elapsed_stl[i]),
            boost::ref(elapsed_lockfree[i])
            ));

    workers.join_all();

//    std::pair<double, double> total_elapsed_stl(0.0, 0.0);
    std::pair<double, double> total_elapsed_lockfree(0.0, 0.0);

    for (boost::uint64_t i = 0; i < elapsed_stl.size(); ++i)
    {
//        total_elapsed_stl.first  += elapsed_stl[i].first;
//        total_elapsed_stl.second += elapsed_stl[i].second;

        total_elapsed_lockfree.first  += elapsed_lockfree[i].first;
        total_elapsed_lockfree.second += elapsed_lockfree[i].second;
    }

    // Print out the results.
//    print_results(vm, total_elapsed_stl, total_elapsed_lockfree);
    print_results(vm, total_elapsed_lockfree);

    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(
    int argc
  , char* argv[]
    )
{
    ///////////////////////////////////////////////////////////////////////////
    // Parse command line.
    variables_map vm;

    options_description cmdline("Usage: serial_lifo_overhead [options]");

    cmdline.add_options()
        ( "help,h"
        , "print out program usage (this message)")

        ( "threads,t"
        , value<boost::uint64_t>(&threads)->default_value(1)
        , "number of threads to use")

        ( "iterations"
        , value<boost::uint64_t>(&iterations)->default_value(2000000)
        , "number of iterations to perform (most be divisible by block size)")

        ( "blocksize"
        , value<boost::uint64_t>(&blocksize)->default_value(10000)
        , "size of each block")

//        ( "delay"
//        , value<boost::uint64_t>(&delay)->default_value(5)
//        , "duration of delay in microseconds")

        ( "no-header"
        , "do not print out the header")
        ;

    store(command_line_parser(argc, argv).options(cmdline).run(), vm);

    notify(vm);

    // Print help screen.
    if (vm.count("help"))
    {
        std::cout << cmdline;
        return 0;
    }

    if (iterations % blocksize)
        throw std::invalid_argument(
            "iterations must be cleanly divisable by blocksize\n");

    if (vm.count("no-header"))
        header = false;

    return app_main(vm);
}

