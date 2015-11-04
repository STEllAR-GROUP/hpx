//  Copyright (c) 2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This must succeed compiling

#include <hpx/hpx_main.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/apply.hpp>

void test(int const& ref) {}
HPX_PLAIN_ACTION(test);

///////////////////////////////////////////////////////////////////////////////
int main()
{
    int val = 0;

    test_action act;
    hpx::apply(act, hpx::find_here(), val);

    return 0;
}

