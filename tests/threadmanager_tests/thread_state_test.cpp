//  Copyright (c) 2008-2009 Anshul Tandon
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#include <iostream>
#include <hpx/hpx.hpp>
#include <boost/detail/lightweight_test.hpp>

using namespace hpx;
using namespace hpx::threads;

thread_state my_gcd (int m, int n, int gcd);
void print_state (thread_state t_s);

typedef hpx::threads::threadmanager_impl<hpx::threads::policies::global_queue_scheduler> threadmanager_type;

int main(int argc, char* argv[])
{
    hpx::util::io_service_pool timer_pool;
    hpx::threads::policies::global_queue_scheduler scheduler;
    hpx::threads::policies::callback_notifier notifier;
    threadmanager_type my_tm(timer_pool, scheduler, notifier);

    threads::thread_init_data data1(boost::bind (my_gcd, 13, 14, 1), "gcd");
    my_tm.register_work(data1);                      // GCD = 1

    threads::thread_init_data data2(boost::bind (my_gcd, 7, 343, 7), "gcd");
    hpx::threads::thread_id_type t_id = 
        my_tm.register_thread(data2, suspended);     // GCD = 7

    threads::thread_init_data data3(boost::bind (my_gcd, 120, 115, 5), "gcd");
    hpx::threads::thread_id_type t2_id = 
        my_tm.register_thread(data3, suspended);   // GCD = 5

    threads::thread_init_data data4(boost::bind (my_gcd, 9, 15, 3), "gcd");
    my_tm.register_work(data4, pending);              // GCD = 3

    BOOST_TEST(my_tm.get_state(t2_id) == suspended);
    my_tm.set_state(t2_id, pending);
    BOOST_TEST(my_tm.get_state(t2_id) == pending);

    BOOST_TEST(my_tm.get_state(t_id) == suspended);
    if (my_tm.get_state(t_id) == pending)
        std::cout << "Error, thread ID invalid" << std::endl;

    if (my_tm.get_state(t_id) == suspended)
        my_tm.set_state(t_id, pending);
    BOOST_TEST(my_tm.get_state(t_id) == pending);

    thread_state t_s = my_tm.get_state(t2_id);
    BOOST_TEST(t_s == pending);
//     print_state (t_s);

    for (int i = 1; i <= 8; ++i) {
        my_tm.run(i);
        while ((t_s = my_tm.get_state(t2_id)) != unknown)
        {
            BOOST_TEST(t_s == pending || t_s == active || t_s == terminated);
        }
        my_tm.stop();
    }

    return boost::report_errors();
}

thread_state my_gcd (int m, int n, int gcd)
{
    int r;
    while(n != 0){
        r = m % n;
        m = n;
        n = r;
    }

    get_self().yield(pending);   // just reschedule

    BOOST_TEST(m == gcd);
//     std::cout << "GCD for the two numbers is: " << m << std::endl;
    return terminated;
}

void print_state (thread_state t_s)
{
    switch (t_s) {
    case unknown:    std::cout << "Unknown" << std::endl << std::flush;    break;
    case init:       std::cout << "Init" << std::endl << std::flush;       break;
    case active:     std::cout << "Active" << std::endl << std::flush;     break;
    case pending:    std::cout << "Pending" << std::endl << std::flush;    break;
    case suspended:  std::cout << "Suspended" << std::endl << std::flush;  break;
    case depleted:   std::cout << "Depleted" << std::endl << std::flush;   break;
    case terminated: std::cout << "Terminated" << std::endl << std::flush; break;
    default:         std::cout << "ERROR!!!!" << std::endl << std::flush;  break;
    }
    return;
}

