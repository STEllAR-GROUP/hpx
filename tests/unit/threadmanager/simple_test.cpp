//  Copyright (c) 2008-2009 Chirag Dekate, Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#include <iostream>

#include <hpx/hpx.hpp>

using namespace hpx;
using namespace std;


threadmanager::thread_state my_gcd(hpx::threadmanager::px_thread_self&, int m, int n);

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    hpx::threadmanager::threadmanager my_tm;

    my_tm.register_work(boost::bind (my_gcd, _1, 13, 14));
    my_tm.register_work(boost::bind (my_gcd, _1, 14, 115));
    my_tm.register_work(boost::bind (my_gcd, _1, 14, 12));
    my_tm.run();
    my_tm.stop();

    return 0;
}

threadmanager::thread_state my_gcd(hpx::threadmanager::px_thread_self& s, int m, int n)
{
    int r;
    while(n != 0){
        r = m%n;
        m = n;
        n = r;
    }
    s.yield(threadmanager::suspended);
    std::cout << "GCD for the two numbers is: " << m << std::endl;
    return threadmanager::terminated;
}
