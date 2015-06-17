//  Copyright (c) 2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test case demonstrates the issue described in #790: wait_for() doesn't
// compile

#include <hpx/hpx.hpp>
#include <hpx/hpx_main.hpp>
#include <hpx/lcos/future.hpp>

int main()
{
    hpx::lcos::future<int> future = hpx::lcos::make_ready_future(0);
    boost::chrono::nanoseconds tn(static_cast<long long>(1000000000LL));
    future.wait_for(tn);

    return 0;
}
