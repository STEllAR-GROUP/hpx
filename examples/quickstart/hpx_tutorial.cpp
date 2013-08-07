////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2013 Andrew Kemp
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/hpx_main.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/iostreams.hpp>

#include <iostream>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
//[hpx_tutorial_section_1
//SECTION 1
// SEE 2.b to find where this function is used!
int addone(int input)
{
    return input + 1;  
}
//] 

//[hpx_tutorial_section_2
//SECTION 2
//2.a
int main()
{
    //2.b
    size_t const os_threads = hpx::get_os_thread_count();
    std::cout << "Program will run on " << os_threads << " cores concurrently.\nIncrease or decrease the number of cores used by passing the -tX argument to the program, where X is the number of threads to run.\n";
    //] 
    //[hpx_tutorial_section_3
    //SECTION 3
    int value = 0;
    std::cout << "This first invocation will be loaded with the value: " << value << "\n";
    hpx::future<int> invocation1 = hpx::async(&addone, value);
    std::cout << "This second invocation will be loaded with the value: " << value << "\n";
    ++value;
    hpx::future<int> invocation2 = hpx::async(&addone, value);

    //] 
    //[hpx_tutorial_section_4
    //SECTION 4
    int returned = invocation1.get();
    std::cout << "The first invocation has returned the value: " << returned << "\n";
    returned = invocation2.get();
    std::cout << "The second invocation has returned the value: " << returned << "\n";
    //] 
    //[hpx_tutorial_section_5
    //SECTION 5
    return 0;
}

//]
