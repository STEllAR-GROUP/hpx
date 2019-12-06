// Copyright (C) 2019 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/include/threadmanager.hpp>
#include <hpx/testing.hpp>

void stackless_thread()
{
    HPX_TEST_NEQ(hpx::threads::get_self_id(), hpx::threads::invalid_thread_id);
}

int main(int argc, char* argv[])
{
    hpx::threads::register_non_suspendable_work_nullary(stackless_thread);
    return hpx::util::report_errors();
}

