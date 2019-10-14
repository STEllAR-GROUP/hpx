//  Copyright (c) 2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This must fail compiling

#include <hpx/hpx_main.hpp>
#include <hpx/include/actions.hpp>

void test (int const* ptr) {}
HPX_PLAIN_ACTION(test);

///////////////////////////////////////////////////////////////////////////////
int main()
{
    int const* ptr = nullptr;

    test_action act;
    hpx::apply(act, hpx::find_here(), ptr);

    return 0;
}

