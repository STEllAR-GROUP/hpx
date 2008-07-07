//  Copyright (c) 2008-2009 Anshul Tandon
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#include <iostream>
#include <conio.h>
#include <hpx/hpx.hpp>

using namespace hpx;
using namespace hpx::threads;

thread_state my_gcd (hpx::threads::thread_self&, int m, int n);
void print_state (thread_state t_s);
void sleep (unsigned int mseconds);

int main(int argc, char* argv[])
{
    hpx::threads::threadmanager my_tm;

    my_tm.register_work(boost::bind (my_gcd, _1, 13, 14));                      // GCD = 1
    hpx::threads::thread_id_type t_id = 
        my_tm.register_work(boost::bind (my_gcd, _1, 7, 343), suspended);       // GCD = 7
    hpx::threads::thread_id_type t2_id = 
        my_tm.register_work(boost::bind (my_gcd, _1, 120, 115), suspended);     // GCD = 5
    my_tm.register_work(boost::bind (my_gcd, _1, 9, 15), pending);              // GCD = 3

    my_tm.set_state(t2_id, pending);

    if (my_tm.get_state(t_id) == pending)
        std::cout << "Error, thread ID invalid" << std::endl;
    
    if (my_tm.get_state(t_id) == suspended)
        my_tm.set_state(t_id, pending);

    thread_state t_s = my_tm.get_state(t2_id);
    print_state (t_s);

    my_tm.run();

//    sleep(1);
    while (my_tm.get_state(t2_id) != unknown)
    {
        print_state (my_tm.get_state(t2_id));
    }

    my_tm.stop();

    return 0;
}

thread_state my_gcd (hpx::threads::thread_self& s, int m, int n)
{
    int r;
    while(n != 0){
        r = m%n;
        m = n;
        n = r;
    }
    s.yield(pending);
    std::cout << "GCD for the two numbers is: " << m << std::endl;
    return terminated;
}

void print_state (thread_state t_s)
{
    switch (t_s)
    {
        case unknown:       std::cout << "Unknown" << std::endl;      break;
        case init:          std::cout << "Init" << std::endl;         break;
        case active:        std::cout << "Active" << std::endl;       break;
        case pending:       std::cout << "Pending" << std::endl;      break;
        case suspended:     std::cout << "Suspended" << std::endl;    break;
        case depleted:      std::cout << "Depleted" << std::endl;     break;
        case terminated:    std::cout << "Terminated" << std::endl;   break;
        default:            std::cout << "ERROR!!!!" << std::endl;    break;
    }
    return;
}

void sleep (unsigned int mseconds)
{
    clock_t goal = mseconds + clock();
    while (goal > clock());
}
