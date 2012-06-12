
//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>

#include <oclm/oclm.hpp>

int main()
{
    oclm::get_platform();
    
    const char src[]
               = "__kernel "
                 "void vecadd(__global int* A, __global int* B, __global int* C)"
                 "{"
                 "  int tid = get_global_id(0);"
                 "  C[tid] = A[tid] + B[tid];"
                 "}"
               ;

    // create a program from source ... possibly a vector of sources ...
    oclm::program p(src);

    {
        // Select a kernel
        oclm::kernel k(p, "vecadd");

        // build a function object out of the kernel
        oclm::function f = k[oclm::global(80), oclm::local(1)];

        std::vector<int> A(80, 1);
        std::vector<int> B(80, 2);
        std::vector<int> C(80, 0);
        
        // create a command queue with a device type and a platform ... context and
        // platform etc is selected in the background ... this will be managed as
        // global state
        oclm::command_queue queue(oclm::device::gpu);


        // asynchronously fire the opencl function on the command queue, the
        // std::vector's will get copied back and forth transparantly, policy classes
        // to come ...
        oclm::event e1 = async(queue, f(A, B, C));

        // wait until everything is completed ...
        //e1.get()

        // sanity check ...
        BOOST_FOREACH(int i, A)
        {
            BOOST_ASSERT(i == 1);
        }

        BOOST_FOREACH(int i, B)
        {
            BOOST_ASSERT(i == 2);
        }

        BOOST_FOREACH(int i, C)
        {
            BOOST_ASSERT(i == 3);
        }
    }
    
    {
        // build a function object out of the kernel
        oclm::function f(p, "vecadd", oclm::global(80), oclm::local(1));

        std::vector<int> A(80, 1);
        std::vector<int> B(80, 2);
        std::vector<int> C(80, 0);
        
        // create a command queue with a device type and a platform ... context and
        // platform etc is selected in the background ... this will be managed as
        // global state
        oclm::command_queue queue(oclm::device::gpu);


        // asynchronously fire the opencl function on the command queue, the
        // std::vector's will get copied back and forth transparantly, policy classes
        // to come ...
        oclm::event e1 = async(queue, f(A, B, C));

        // wait until everything is completed ...
        //e1.get()

        // sanity check ...
        BOOST_FOREACH(int i, A)
        {
            BOOST_ASSERT(i == 1);
        }

        BOOST_FOREACH(int i, B)
        {
            BOOST_ASSERT(i == 2);
        }

        BOOST_FOREACH(int i, C)
        {
            BOOST_ASSERT(i == 3);
        }
    }
}
