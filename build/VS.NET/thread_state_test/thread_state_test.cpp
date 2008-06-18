//  Copyright (c) 2008-2009 Anshul Tandon
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#include <iostream>
#include <hpx/hpx.hpp>

using namespace hpx;
using namespace hpx::threadmanager;

thread_state my_gcd(hpx::threadmanager::px_thread_self&, int m, int n);
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    hpx::threadmanager::threadmanager my_tm;

    my_tm.register_work(boost::bind (my_gcd, _1, 13, 14));                      // GCD = 1
    hpx::threadmanager::px_thread::thread_id_type t_id = 
        my_tm.register_work(boost::bind (my_gcd, _1, 120, 115), suspended);     // GCD = 5
    my_tm.register_work(boost::bind (my_gcd, _1, 9, 15), pending);              // GCD = 3

    my_tm.set_state(t_id, pending);

    my_tm.run();
    my_tm.wait();
    my_tm.stop();

    return 0;
}

thread_state my_gcd(hpx::threadmanager::px_thread_self& s, int m, int n)
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
