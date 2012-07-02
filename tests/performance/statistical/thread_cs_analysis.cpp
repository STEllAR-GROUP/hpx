//  Copyright (c)      2012 Daniel Kogler
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "statstd.hpp"
#include <hpx/include/threadmanager.hpp>
#include <hpx/util/lightweight_test.hpp>
#include <boost/assign/std/vector.hpp>
#include <boost/lexical_cast.hpp>

void loop_function(uint64_t iters){
    volatile double bigcount = iters;
    for(uint64_t i = 0; i < iters; i++){
        bigcount*=2;
        bigcount/=2;
    }
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

//this runs a series of tests for a packaged_action.apply()
void run_tests(uint64_t, int);

///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm){
    uint64_t num = vm["iterations"].as<uint64_t>();
    int threads = vm["hpx-threads"].as<int>();
    csv = (vm.count("csv") ? true : false);
    run_tests(num, threads);
    return hpx::finalize();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[]){
    // Configure application-specific options.
    options_description
        desc_commandline("usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ("iterations,N",
            boost::program_options::value<uint64_t>()
                ->default_value(5000000),
            "number of iterations the loop function will iterate over")
        ("hpx-threads,H",
            boost::program_options::value<int>()->default_value(2),
            "number of simultaneous hpx threads running")
        ("csv",
            "output results as csv "
            "(format:count,mean,accurate mean,variance,min,max)");

    // Initialize and run HPX
    return hpx::init(desc_commandline, argc, argv);
}

///////////////////////////////////////////////////////////////////////////////

//measure how long it takes to spawn threads with a simple argumentless function 
void run_tests(uint64_t num, int threads){
    double time1, time2;
    int i;
    vector<hpx::thread> funcs;
    //first measure how long it takes to spawn threads
    //then measure how long it takes to join them back together
    printf("\nNOTE: for now, this benchmark is intended to obtain the \n"
            "       percentage of total runtime spent on context switching when\n"
            "       running on a single OS thread\n\n");

    high_resolution_timer t;
    for(i = 0; i < threads; i++)
        loop_function(num);
    time1 = t.elapsed();

    t.restart();
    for(i = 0; i < threads; ++i)
        funcs.push_back(hpx::thread(&loop_function, num));
    for(i = 0; i < threads; ++i)
        funcs[i].join();
    time2 = t.elapsed();

    printf("Executing the function %d times sequentially yields a total time of "
        "%f s\n", threads, time1);
    printf("Executing the function %d times simultaneously yields a total time of "
        "%f s\n\n", threads, time2);

    printf("Estimated time spent context switching is %f s\n\n", time2-time1);
    printf("Estimated percentage of time spent context switching is %% %f\n\n",
        (1-time1/(time2))*100);
}

