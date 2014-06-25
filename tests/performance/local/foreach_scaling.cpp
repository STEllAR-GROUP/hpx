//  Copyright (c) 2014 Grant Mercer
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/util/high_resolution_timer.hpp>
#include <hpx/include/algorithm.hpp>
#include <hpx/stl/execution_policy.hpp>
#include <hpx/include/iostreams.hpp>
#include "worker_timed.hpp"

#include <stdexcept>

#include <boost/format.hpp>
#include <boost/chrono.hpp>
#include <boost/cstdint.hpp>
enum {
    SECONDS=0,
    MILLISECONDS,
    MICROSECONDS
};

//constants to convert seconds
const int c_MICROSECONDS = 1000000;
const int c_MILLISECONDS = 1000;

hpx::util::high_resolution_timer walltime;

int delay;
int test_count;

double measure_sequential_foreach(std::size_t size, int measure)
{
    std::vector<std::size_t> data_representation(size);
    std::iota(boost::begin(data_representation),
        boost::end(data_representation),
        std::rand());

    walltime.restart();

    //invoke sequential for_each
    hpx::parallel::for_each(hpx::parallel::seq,
        boost::begin(data_representation),
        boost::end(data_representation),
        [&size](std::size_t& v){
            worker_timed(delay);
            v=40;
        });

    double duration = walltime.elapsed();
    switch(measure) {
    case MILLISECONDS:
        return duration * c_MILLISECONDS;
    case MICROSECONDS:
        return duration * c_MICROSECONDS;
    default:
        break;
    }
    return duration;
}

double measure_parallel_foreach(std::size_t size, int measure)
{
    std::vector<std::size_t> data_representation(size);
    std::iota(boost::begin(data_representation),
        boost::end(data_representation),
        std::rand());

    walltime.restart();

    //invoke parallel for_each
    hpx::parallel::for_each(hpx::parallel::par,
        boost::begin(data_representation),
        boost::end(data_representation),
        [&size](std::size_t& v){
            worker_timed(delay);
            v=40;
        });

    double duration = walltime.elapsed();

  
    switch(measure) {
    case MILLISECONDS:
        return duration * c_MILLISECONDS;
    case MICROSECONDS:
        return duration * c_MICROSECONDS;
    default:
        break;
    }
    return duration;
}

double average_out_parallel(std::size_t vector_size, int mtime)
{
    double total_time =0;
    //average out 100 executions to avoid varying results
    for(auto i=0; i < test_count; i++)
        total_time += measure_parallel_foreach(vector_size, mtime);
    return total_time/(double)(test_count);
}

double average_out_sequential(std::size_t vector_size, int mtime)
{
    double total_time =0;
    //average out 100 executions to avoid varying results
    for(auto i=0; i<test_count; i++)
        total_time += measure_sequential_foreach(vector_size, mtime);
    return total_time/(double)(test_count);
}

int hpx_main(boost::program_options::variables_map& vm)
{
    //pull values from cmd
    std::size_t vector_size = vm["vector_size"].as<std::size_t>();
    int mtime = vm["mtime"].as<int>();
    delay = vm["work_delay"].as<int>();
    test_count = vm["test_count"].as<int>();
    
    //verify that input is within domain of program
    if(test_count == 0 || test_count < 0) {
        hpx::cout << "test_count cannot be zero or negative...\n" << hpx::flush;
    } else if (delay < 0) {
        hpx::cout << "delay cannot be a negative number...\n" << hpx::flush;
    } else if (mtime > 2 || mtime < 0) {
        hpx::cout << "invalid mtime range, mtime cannot be greater than 2 or negative...\n" << hpx::flush;
    } else {

    //results
    double par_time = average_out_parallel(vector_size, mtime);
    double seq_time = average_out_sequential(vector_size, mtime); 

    hpx::cout << boost::format("test..\n");
    
    //print results(Formatted). Setw(x) assures that all output is right justified
    hpx::cout << std::left << "----------------Parameters-----------------\n"
        << std::left << "Vector size: " << std::right << std::setw(30) << vector_size << "\n"
        << std::left << "Number of tests" << std::right << std::setw(28) << test_count << "\n"
        << std::left << "Delay per iteration(MicroSeconds)" << std::right << std::setw(10) << delay << "\n"
        << std::left << "Display time in: " << hpx::flush;
    if(mtime==0) hpx::cout << std::right << std::setw(27) << "Seconds\n" << hpx::flush;
    if(mtime==1) hpx::cout << std::right << std::setw(27) << "Miliseconds\n" << hpx::flush;
    if(mtime==2) hpx::cout << std::right << std::setw(27) << "Microseconds\n" << hpx::flush;
    hpx::cout << "------------------Average------------------\n"
    << std::left << "Average parallel execution time : " << std::right << std::setw(9) << par_time << "\n"
    << std::left << "Average sequential execution time: " << std::right << std::setw(8) << seq_time << "\n" << hpx::flush;
    hpx::cout << "---------Execution Time Difference---------\n"
    << std::left << "Parallel <-> Sequential: " << std::right << std::setw(18) << par_time - seq_time << "\n"
    << std::left << "Parallel Scale: " << std::right  << std::setw(27) << ((double)seq_time / par_time) << "\n" << hpx::flush;
    }

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    //initialize program
    std::vector<std::string> cfg;
    cfg.push_back("hpx.os_threads=" +
        boost::lexical_cast<std::string>(hpx::threads::hardware_concurrency()));
    boost::program_options::options_description cmdline("usage: " HPX_APPLICATION_STRING " [options]");
    
    cmdline.add_options()
        ( "vector_size"
        , boost::program_options::value<std::size_t>()->default_value(1000)
        , "size of vector")

        ( "mtime"
        , boost::program_options::value<int>()->default_value(0)
        , "(0)Seconds (1)Milliseconds (2)Microseconds")

        ("work_delay"
        , boost::program_options::value<int>()->default_value(1)
        , "loop delay per element in Microseconds")

        ("test_count"
        , boost::program_options::value<int>()->default_value(100)
        , "number of tests to be averaged")
        ;

    return hpx::init(cmdline, argc, argv, cfg);
}
