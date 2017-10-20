//  Copyright (c) 2017 Christopher HInx
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/include/thread_executors.hpp>
#include <hpx/util/lightweight_test.hpp>

int main()
{
    bool caught_exception = false;
    try
    {
        hpx::threads::executors::local_priority_queue_os_executor exec(1);
    }
    catch (...)
    {
        caught_exception = true;
    }
    HPX_TEST(caught_exception);

    return hpx::util::report_errors();
}
