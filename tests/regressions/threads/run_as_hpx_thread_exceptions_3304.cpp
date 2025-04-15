//  Copyright (c) 2018-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/condition_variable.hpp>
#include <hpx/exception.hpp>
#include <hpx/functional.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/mutex.hpp>
#include <hpx/runtime_local/run_as_hpx_thread.hpp>

#include <functional>
#include <mutex>
#include <thread>

void hpx_thread_func()
{
    HPX_THROW_EXCEPTION(hpx::error::invalid_status, "hpx_thread_func", "test");
}

int main(int argc, char** argv)
{
    hpx::manage_runtime rt;

    rt.start(argc, argv);

    bool exception_caught = false;
    try
    {
        hpx::run_as_hpx_thread(&hpx_thread_func);
        HPX_TEST(false);    // this should not be executed
    }
    catch (...)
    {
        exception_caught = true;
    }
    HPX_TEST(exception_caught);

    return rt.stop();
}
