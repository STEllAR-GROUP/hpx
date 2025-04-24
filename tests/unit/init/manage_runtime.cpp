//  Copyright (c) 2025 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/assert.hpp>
#include <hpx/manage_runtime.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/program_options.hpp>
#include <hpx/runtime_local/run_as_hpx_thread.hpp>

hpx::runtime* hpx_thread_func()
{
    return hpx::get_runtime_ptr();
}

int main(int argc, char* argv[])
{
    // Start the runtime in whatever mode is available
    {
        hpx::manage_runtime rt;

        HPX_TEST(hpx::get_runtime_ptr() == nullptr);
        HPX_TEST(rt.start(argc, argv));

        HPX_TEST(
            hpx::get_runtime_ptr() == hpx::run_as_hpx_thread(hpx_thread_func));

        HPX_TEST(rt.stop() == 0);
        HPX_TEST(hpx::get_runtime_ptr() == nullptr);
    }

    // Also when the init parameters struct is explicitly passed.
    {
        hpx::manage_runtime rt;

        HPX_TEST(hpx::get_runtime_ptr() == nullptr);

        hpx::init_params iparams;
        HPX_TEST(rt.start(argc, argv, iparams));

        HPX_TEST(
            hpx::get_runtime_ptr() == hpx::run_as_hpx_thread(hpx_thread_func));

        HPX_TEST(rt.stop() == 0);
        HPX_TEST(hpx::get_runtime_ptr() == nullptr);
    }

    // The local runtime should always be possible to start (even though not
    // all functionality may work)
    {
        hpx::manage_runtime rt;

        HPX_TEST(hpx::get_runtime_ptr() == nullptr);

        hpx::init_params iparams;
        iparams.mode = hpx::runtime_mode::local;
        HPX_TEST(rt.start(argc, argv, iparams));

        HPX_TEST(
            hpx::get_runtime_ptr() == hpx::run_as_hpx_thread(hpx_thread_func));

        HPX_TEST(rt.stop() == 0);
        HPX_TEST(hpx::get_runtime_ptr() == nullptr);
    }

    return hpx::util::report_errors();
}
