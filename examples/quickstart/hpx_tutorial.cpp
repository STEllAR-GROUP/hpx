////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2013 Andrew Kemp
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/hpx_init.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/util.hpp>
#include <hpx/include/lcos.hpp>

#include <iostream>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
//[SECTION_1
// SEE 3.c to find where this function is used!
int addone(int input)
{
    return input + 1;  
}
//]
//[SECTION_2
int main(int argc, char *argv[])
{
    return hpx::init(argc, argv);
}
//]
//[SECTION_3

int hpx_main()
{
    // 3.a
    size_t const os_threads = hpx::get_os_thread_count();
    std::cout << "Program will run on " << os_threads << " cores concurrently.\nIncrease or decrease the number of cores used by passing the -tX argument to the program, where X is the number of threads to run.\n";
    
    int value = 0;
    std::cout << "The thread will be loaded with the value: " << value << "\n";
    // 3.b
    hpx::future<int> thread = hpx::async(&addone, value);
    // 3.c
    int returned = thread.get();
    std::cout << "The thread has returned the value: " << returned << "\n";
    return hpx::finalize(); // Handles HPX shutdown
}
//]
