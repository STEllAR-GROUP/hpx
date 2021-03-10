//  Copyright (c) 2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This must succeed compiling

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx_main.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/apply.hpp>
#include <hpx/include/runtime.hpp>

#include <utility>

void test(int&& ref) {}
HPX_PLAIN_ACTION(test);

///////////////////////////////////////////////////////////////////////////////
int main()
{
    int val = 0;

    test_action act;
    hpx::apply(act, hpx::find_here(), std::move(val));

    return 0;
}
#endif
