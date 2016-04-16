//  Copyright (c) 2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test case demonstrates the issue described in #1702: Shared_mutex does
// not compile with no_mutex cond_var

#include <hpx/hpx.hpp>
#include <hpx/hpx_main.hpp>
#include <hpx/lcos/local/shared_mutex.hpp>

#include <boost/thread/locks.hpp>

#include <mutex>

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
        boost::shared_lock<shared_mutex_type> l(mtx);
        int i = data;
        HPX_UNUSED(i);
    }

    return 0;
}
