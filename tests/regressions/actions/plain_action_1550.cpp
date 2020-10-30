//  Copyright (c) 2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Verify that #1550 was properly fixed (Properly fix HPX_DEFINE_*_ACTION macros)

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx_init.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/async.hpp>
#include <hpx/modules/testing.hpp>
#include <utility>

bool called_test_action = false;
bool called_t = false;

namespace mynamespace
{
    void test()
    {
        called_test_action = true;
    }

    HPX_DEFINE_PLAIN_ACTION(test);

    static auto t =
        hpx::actions::lambda_to_action(
        []()
        {
            called_t = true;
        });
}

typedef mynamespace::test_action mynamespace_test_action;

HPX_REGISTER_ACTION(mynamespace_test_action);

int hpx_main()
{
    {
        {
            typedef mynamespace_test_action func;
            hpx::async<func>(hpx::find_here()).get();
        }

        HPX_TEST(called_test_action);
    }

    // Same test with lambdas
    // action lambdas inhibit undefined behavior...
#if !defined(HPX_HAVE_SANITIZERS)
    {
        {
            hpx::async(std::move(mynamespace::t),hpx::find_here()).get();
        }

        HPX_TEST(called_t);
    }
#endif

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ(hpx::init(argc, argv), 0);
    return hpx::util::report_errors();
}
#endif
