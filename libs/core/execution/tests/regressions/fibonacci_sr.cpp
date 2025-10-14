//  Copyright (c) 2022 Steven R. Brandt
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

// Clang V11 ICE's on this test
#if !defined(HPX_CLANG_VERSION) || (HPX_CLANG_VERSION / 10000) != 11

#include <hpx/execution.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>

namespace ex = hpx::execution::experimental;
namespace tt = hpx::this_thread::experimental;

ex::any_sender<int> fib3(int n)
{
    if (n < 2)
    {
        return ex::just(n);
    }

    return ex::when_all(fib3(n - 1), fib3(n - 2)) |
        ex::then([](int n1, int n2) { return n1 + n2; });
}

int hpx_main()
{
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    HPX_TEST_EQ(hpx::get<0>(*tt::sync_wait(fib3(15))), 610);
    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ_MSG(hpx::local::init(hpx_main, argc, argv), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}

#else

int main(int, char*[])
{
    return 0;
}

#endif
