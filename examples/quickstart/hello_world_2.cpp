//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

///////////////////////////////////////////////////////////////////////////////
// The purpose of this example is to initialize the HPX runtime explicitly and
// execute a HPX-thread printing "Hello World!" once. That's all.

//[hello_world_2_getting_started

#include <hpx/config.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/iostream.hpp>
#include <hpx/parallel/algorithms/for_each.hpp>

#ifdef HPX_HAVE_CONTRACTS

// Currently testing with C++23 draft (value 202302L).
// When C++26 is officially published, this macro value will likely
// update to 202602L or similar. Review this check at that time.
  #if __cplusplus < 202302L
    #pragma message("Warning: Contracts require C++23 or later. Contracts disabled.")
    #define HPX_PRECONDITION(expr)
  #else
    #define HPX_PRECONDITION(expr) pre((expr))
  #endif
#else
  #define HPX_PRECONDITION(expr)
#endif



int hpx_main(int, char*[])
HPX_PRECONDITION(false)
{
    // Say hello to the world!
    hpx::cout << "Hello World! time for contracts using def!!!1!!\n" << std::flush;
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    return hpx::init(argc, argv);
}


// void for_each_bridge(auto &data)
// pre(data.end()>data.begin())
// {

//     hpx::for_each(hpx::execution::seq, data.end(), data.begin(),
//         [](int& value) { value *= 2; });
// }

// int hpx_main(int, char*[])
// {
//     // Say hello to the world!
//     hpx::cout << "Hello World!\n" << std::flush;

//     std::vector<int> data = {1, 2, 3, 4, 5};
//     hpx::cout << (data.end()>data.begin()) << std::flush;
     
//     for_each_bridge(data);
    


//     hpx::cout << "Modified data: ";
//     for (const auto& value : data)
//     {
//         hpx::cout << value << " ";
//     }
//     hpx::cout << "\n" << std::flush;



//     return hpx::finalize();
// }

// int main(int argc, char* argv[])
// {
//     return hpx::init(argc, argv);
// }
//]
