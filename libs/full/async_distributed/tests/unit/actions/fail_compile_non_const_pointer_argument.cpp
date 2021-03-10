//  Copyright (c) 2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This must fail compiling

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx_main.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/runtime.hpp>

void test(int* ptr) {}
HPX_PLAIN_ACTION(test);

///////////////////////////////////////////////////////////////////////////////
int main()
{
    int* ptr = nullptr;

    test_action act;
    hpx::apply(act, hpx::find_here(), ptr);

    return 0;
}
#endif
