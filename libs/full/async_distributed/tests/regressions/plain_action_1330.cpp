//  Copyright (c) 2014 Hartmut Kaiser
//  Copyright (c) 2014 Martin Stumpf
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/futures/future.hpp>
#include <hpx/hpx_start.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/runtime.hpp>
#include <hpx/iostream.hpp>
#include <hpx/modules/testing.hpp>

#include <utility>

bool called_test_action = false;
bool called_t = false;

namespace mynamespace {
    void test()
    {
        called_test_action = true;
    }

    HPX_DEFINE_PLAIN_ACTION(test, test_action);

    static auto t = hpx::actions::lambda_to_action([]() { called_t = true; });
}    // namespace mynamespace

HPX_REGISTER_ACTION(mynamespace::test_action, mynamespace_test_action);

// hpx_main, is the actual main called by hpx
int hpx_main()
{
    {
        using func = mynamespace::test_action;
        hpx::async<func>(hpx::find_here()).get();
        HPX_TEST(called_test_action);
    }

    // Same test with lambdas
    // action lambdas inhibit undefined behavior...
#if !defined(HPX_HAVE_SANITIZERS)
    {
        hpx::async(std::move(mynamespace::t), hpx::find_here()).get();
        HPX_TEST(called_t);
    }
#endif

    // End the program
    return hpx::finalize();
}

// Main, initializes HPX
int main(int argc, char* argv[])
{
    // initialize HPX, run hpx_main
    hpx::start(argc, argv);

    // wait for hpx::finalize being called
    HPX_TEST_EQ(hpx::stop(), 0);
    return hpx::util::report_errors();
}
#endif
