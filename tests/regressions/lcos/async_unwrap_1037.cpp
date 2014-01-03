//  Copyright 2013 (c) Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test case demonstrates the issue described in #1037:
// implicit unwrapping of futures in async broken

#include <hpx/hpx.hpp>
#include <hpx/hpx_main.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/async.hpp>
#include <hpx/util/lightweight_test.hpp>

hpx::future<int> f(int i)
{
    if(i == 0)
    {
        return hpx::async(hpx::util::bind(f, i+1));
    }

    return hpx::make_ready_future<int>(i);
}

int main()
{
    HPX_TEST_EQ(f(0).get(), 1);
    return hpx::util::report_errors();
}
