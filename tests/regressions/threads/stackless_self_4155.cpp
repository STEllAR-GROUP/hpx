// Copyright (C) 2019 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/local/init.hpp>
#include <hpx/local/thread.hpp>
#include <hpx/modules/testing.hpp>

void stackless_thread()
{
    HPX_TEST_NEQ(hpx::threads::get_self_id(), hpx::threads::invalid_thread_id);
}

int hpx_main()
{
    hpx::threads::thread_init_data data(
        hpx::threads::make_thread_function_nullary(stackless_thread),
        "stackless_thread", hpx::threads::thread_priority::default_,
        hpx::threads::thread_schedule_hint(),
        hpx::threads::thread_stacksize::nostack);
    hpx::threads::register_work(data);
    return hpx::local::finalize();
}

int main(int argc, char** argv)
{
    hpx::local::init(hpx_main, argc, argv);

    return hpx::util::report_errors();
}
