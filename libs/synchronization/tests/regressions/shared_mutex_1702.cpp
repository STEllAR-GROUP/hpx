//  Copyright (c) 2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test case demonstrates the issue described in #1702: Shared_mutex does
// not compile with no_mutex cond_var

#include <hpx/hpx.hpp>
#include <hpx/hpx_main.hpp>
#include <hpx/synchronization/shared_mutex.hpp>

#include <mutex>
#include <shared_mutex>

int main()
{
    typedef hpx::lcos::local::shared_mutex shared_mutex_type;

    int data = 0;
    shared_mutex_type mtx;

    {
        std::unique_lock<shared_mutex_type> l(mtx);
        data = 42;
    }

    {
        std::shared_lock<shared_mutex_type> l(mtx);
        int i = data;
        HPX_UNUSED(i);
    }

    return 0;
}
