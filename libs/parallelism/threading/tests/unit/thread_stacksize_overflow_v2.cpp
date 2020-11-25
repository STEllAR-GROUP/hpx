// Copyright (C) 2012 Hartmut Kaiser
// Copyright (C) 2017 Christopher Taylor
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx_main.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/runtime.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/threading_base/thread_data.hpp>

#include <cstring>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
void test_small_stacksize()
{
    HPX_TEST(hpx::threads::get_self_ptr());
    // verify that sufficient stack has been allocated
    HPX_TEST_EQ(hpx::threads::get_ctx_ptr()->get_stacksize(),
        hpx::get_runtime().get_config().get_stack_size(
            hpx::threads::thread_stacksize::small_));

    // allocate HPX_SMALL_STACK_SIZE - HPX_THREADS_STACK_OVERHEAD memory on the stack
    char array[HPX_SMALL_STACK_SIZE * HPX_THREADS_STACK_OVERHEAD];

    // do something to that array
    std::memset(array, '\0', sizeof(array));
}
HPX_DECLARE_ACTION(test_small_stacksize, test_small_stacksize_action)
HPX_ACTION_USES_SMALL_STACK(test_small_stacksize_action)
HPX_PLAIN_ACTION(test_small_stacksize, test_small_stacksize_action)

int main()
{
    int* i = nullptr;
    int x = (*i);
    return hpx::util::report_errors();
}
#endif
