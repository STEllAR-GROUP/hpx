//  (C) Copyright 2013-2015 Steven R. Brandt
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#include <hpx/lcos/local/composable_guard.hpp>
#include <hpx/hpx_init.hpp>
#include <boost/atomic.hpp>
#include <iostream>

// This program illustrates the use of composable guards to perform
// a computation that might otherwise rely on mutexes. In fact, guards
// are like mutexes, but with the distinction that they are locked for
// you (at the start of a task), and unlocked for you.
//
// In the example below, we use guards to manage incrementing the variables
// i1 and i2. We don't need to use atomics for this, but it makes the
// load, increment, and store steps more explicit. At the end, in order to
// pass the test and get the correct total, every compare_exchange must
// succeed.
//
// Because guards don't add locking structures to the current environment
// during a calculation, they can never deadlock. Guards use only a
// few atomic operations to perform their operaions, and never cause a
// task to block, so they should be quite fast.
//
// To use guards, call the run_guarded() method, supplying it with
// a guard or guard set, and a task to be performed.

typedef boost::atomic<int> int_atomic;
int_atomic i1(0), i2(0);
hpx::lcos::local::guard_set guards;
boost::shared_ptr<hpx::lcos::local::guard> l1(new hpx::lcos::local::guard());
boost::shared_ptr<hpx::lcos::local::guard> l2(new hpx::lcos::local::guard());

void incr1() {
    // implicitly lock l1
    int tmp = i1.load();
    i1.compare_exchange_strong(tmp,tmp+1);
    // implicitly unlock l1
}
void incr2() {
    // implicitly lock l2
    int tmp = i2.load();
    i2.compare_exchange_strong(tmp,tmp+1);
    // implicitly unlock l2
}
void both() {
    // implicitly lock l1 and l2
    int tmp = i1.load();
    i1.compare_exchange_strong(tmp,tmp+1);
    tmp = i2.load();
    i2.compare_exchange_strong(tmp,tmp+1);
    // implicitly unlock l1 and l2
}

int increments = 3000;

void check() {
    if(2*increments == i1 && 2*increments == i2) {
        std::cout << "Test passed" << std::endl;
    } else {
        std::cout << "Test failed: i1=" << i1 << " i2=" << i2 << std::endl;
    }
}

int hpx_main(boost::program_options::variables_map& vm) {
    if (vm.count("increments"))
        increments = vm["increments"].as<int>();

    // create the guard set
    guards.add(l1);
    guards.add(l2);

    for(int i=0;i<increments;i++) {
        // spawn 3 asynchronous tasks
        run_guarded(guards,both);
        run_guarded(*l1,incr1);
        run_guarded(*l2,incr2);
    }

    run_guarded(guards,check);
    return hpx::finalize();
}

int main(int argc, char* argv[]) {
    boost::program_options::options_description
       desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ("increments,n", boost::program_options::value<int>()->default_value(3000),
            "the number of times to increment the counters")
        ;

    // Initialize and run HPX
    return hpx::init(desc_commandline, argc, argv);
}
