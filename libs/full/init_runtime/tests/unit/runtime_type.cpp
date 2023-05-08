//  Copyright (c) 2018 Mikael Simberg
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/program_options.hpp>
#include <hpx/runtime.hpp>

#include <utility>

static bool ran_hpx_main;

// forward declare only (is defined in hpx_init)
int hpx_main(hpx::program_options::variables_map&);

int hpx_main()
{
    ran_hpx_main = true;
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // The default should always start the runtime in whatever mode is
    // available
    {
        ran_hpx_main = false;
        hpx::init(argc, argv);
        HPX_TEST(ran_hpx_main);
    }

    // Also when the init parameters struct is explicitly passed.
    {
        hpx::init_params iparams;
        ran_hpx_main = false;
        hpx::init(argc, argv, iparams);
        HPX_TEST(ran_hpx_main);
    }

    // The local runtime should always be possible to start (even though not
    // all functionality may work)
    {
        hpx::init_params iparams;
        iparams.mode = hpx::runtime_mode::local;
        ran_hpx_main = false;
        hpx::init(argc, argv, iparams);
        HPX_TEST(ran_hpx_main);
    }

    // Back compatibility test
    {
        hpx::init_params iparams;
        iparams.mode = hpx::runtime_mode::local;
        ran_hpx_main = false;

        hpx::function<int(hpx::program_options::variables_map&)> func =
            static_cast<hpx::hpx_main_type>(::hpx_main);

        hpx::init(func, argc, argv, iparams);
        HPX_TEST(ran_hpx_main);
    }

    {
        hpx::init_params iparams;
        iparams.mode = hpx::runtime_mode::local;
        ran_hpx_main = false;

        std::function<int(hpx::program_options::variables_map&)> func =
            static_cast<hpx::hpx_main_type>(::hpx_main);

        hpx::init(func, argc, argv, iparams);
        HPX_TEST(ran_hpx_main);
    }

    {
        hpx::init_params iparams;
        iparams.mode = hpx::runtime_mode::local;
        ran_hpx_main = false;

        std::function<int(hpx::program_options::variables_map&)> func =
            static_cast<hpx::hpx_main_type>(::hpx_main);

        hpx::init(std::move(func), argc, argv, iparams);
        HPX_TEST(ran_hpx_main);
    }

    // The distributed runtime (i.e. any non-runtime_mode::local mode) can only
    // be started when the distributed runtime has been enabled
    {
        hpx::init_params iparams;
        iparams.mode = hpx::runtime_mode::console;

        ran_hpx_main = false;
        bool caught_exception = false;

        try
        {
            hpx::init(argc, argv, iparams);
        }
        catch (...)
        {
            caught_exception = true;
        }

#if defined(HPX_HAVE_DISTRIBUTED_RUNTIME)
        HPX_TEST(ran_hpx_main);
        HPX_TEST(!caught_exception);
#else
        HPX_TEST(!ran_hpx_main);
        HPX_TEST(caught_exception);
#endif
    }

    return hpx::util::report_errors();
}
