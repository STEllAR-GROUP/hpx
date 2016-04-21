//  (C) Copyright 2013-2015 Steven R. Brandt
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#include <hpx/hpx.hpp>
#include <hpx/lcos/local/composable_guard.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/lightweight_test.hpp>
#include <hpx/hpx_init.hpp>
#include <boost/atomic.hpp>
#include <boost/shared_ptr.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <stdlib.h>

typedef boost::atomic<int> int_atomic;
int_atomic i1(0), i2(0);
hpx::lcos::local::guard_set guards;
boost::shared_ptr<hpx::lcos::local::guard> l1(new hpx::lcos::local::guard());
boost::shared_ptr<hpx::lcos::local::guard> l2(new hpx::lcos::local::guard());

void incr1() {
    // implicitly lock l1
    int tmp = i1.load();
    HPX_TEST(i1.compare_exchange_strong(tmp,tmp+1));
    // implicitly unlock l1
}
void incr2() {
    // implicitly lock l2
    int tmp = i2.load();
    HPX_TEST(i2.compare_exchange_strong(tmp,tmp+1));
    // implicitly unlock l2
}
void both() {
    // implicitly lock l1 and l2
    int tmp = i1.load();
    HPX_TEST(i1.compare_exchange_strong(tmp,tmp+1));
    tmp = i2.load();
    HPX_TEST(i2.compare_exchange_strong(tmp,tmp+1));
    // implicitly unlock l1 and l2
}

int increments = 3000;


void check()
{
    HPX_TEST(2*increments == i1 && 2*increments == i2);
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

    boost::function<void()> check_func = hpx::util::bind(check);
    run_guarded(guards,check_func);
    return hpx::finalize();
}

int main(int argc, char* argv[]) {
    boost::program_options::options_description
       desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ("increments,n", boost::program_options::value<int>()->default_value(3000),
            "the number of times to increment the counters")
        ;

    // We force this test to use several threads by default.
    using namespace boost::assign;
    std::vector<std::string> cfg;
    cfg += "hpx.os_threads=" +
        std::to_string(hpx::threads::hardware_concurrency());

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(desc_commandline, argc, argv, cfg), 0,
      "HPX main exited with non-zero status");
    return hpx::util::report_errors();
}
