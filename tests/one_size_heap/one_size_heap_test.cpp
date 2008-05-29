//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#define HPX_USE_ONESIZEHEAPS
#include <hpx/util/one_size_heap_list.hpp>

class test
{
public:
    test() : dummy(0) {}
    
    HPX_DECLARE_ONE_SIZE_PRIVATE_HEAP();    // use one size heaps

private:
    int dummy;
};

// define the one size heap to use
HPX_IMPLEMENT_ONE_SIZE_PRIVATE_HEAP_LIST(
    hpx::util::one_size_heap_allocators::mallocator, test);
    
int main()
{
    test* t = new test;
    delete t;
    return 0;
}
