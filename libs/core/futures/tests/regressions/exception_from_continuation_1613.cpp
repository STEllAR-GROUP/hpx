//  Copyright (c) 2015-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test case demonstrates the issue described in #1613: Dataflow causes
// stack overflow

#include <hpx/future.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>

#include <atomic>
#include <cstddef>
#include <vector>

#define NUM_FUTURES std::size_t(2 * HPX_CONTINUATION_MAX_RECURSION_DEPTH)

void test_exception_from_continuation1()
{
    hpx::promise<void> p;
    hpx::future<void> f1 = p.get_future();

    hpx::future<void> f2 = f1.then([](hpx::future<void>&& f1) {
        HPX_TEST(f1.has_value());
        HPX_THROW_EXCEPTION(
            hpx::error::invalid_status, "lambda", "testing exceptions");
    });

    p.set_value();
    f2.wait();

    HPX_TEST(f2.has_exception());
}

void test_exception_from_continuation2()
{
    hpx::promise<void> p;

    std::vector<hpx::shared_future<void>> results;
    results.reserve(NUM_FUTURES + 1);

    std::atomic<std::size_t> recursion_level(0);
    std::atomic<std::size_t> exceptions_thrown(0);

    results.push_back(p.get_future());
    for (std::size_t i = 0; i != NUM_FUTURES; ++i)
    {
        results.push_back(
            results.back().then([&](hpx::shared_future<void>&& f) {
                ++recursion_level;

                f.get();    // rethrow, if has exception

                ++exceptions_thrown;
                HPX_THROW_EXCEPTION(
                    hpx::error::invalid_status, "lambda", "testing exceptions");
            }));
    }

    // make futures ready in backwards sequence
    hpx::post([&p]() { p.set_value(); });
    HPX_TEST(hpx::wait_all_nothrow(results));

    HPX_TEST_EQ(recursion_level.load(), NUM_FUTURES);
    HPX_TEST_EQ(exceptions_thrown.load(), std::size_t(1));

    // first future is the only one which does not hold exception
    HPX_TEST(!results[0].has_exception());

    for (std::size_t i = 1; i != results.size(); ++i)
    {
        HPX_TEST(results[i].has_exception());
    }
}

int hpx_main()
{
    test_exception_from_continuation1();
    test_exception_from_continuation2();

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    hpx::local::init(hpx_main, argc, argv);
    return hpx::util::report_errors();
}
