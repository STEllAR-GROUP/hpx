//  Copyright (c) 2020 Giannis Gonidelis
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/algorithm.hpp>
#include <hpx/init.hpp>
#include <hpx/iterator_support/tests/iter_sent.hpp>
#include <hpx/modules/testing.hpp>

#include <cstdint>

void myfunction(std::int64_t) {}

void test_invoke_projected()
{
    iterator<std::int64_t> iter = hpx::ranges::for_each(hpx::execution::seq,
        iterator<std::int64_t>{0}, sentinel<std::int64_t>{100}, myfunction);

    HPX_TEST_EQ(*iter, std::int64_t(100));

    iter = hpx::ranges::for_each(hpx::execution::par, iterator<std::int64_t>{0},
        sentinel<std::int64_t>{100}, myfunction);

    HPX_TEST_EQ(*iter, std::int64_t(100));
}

void test_begin_end_iterator()
{
    iterator<std::int64_t> iter = hpx::ranges::for_each(hpx::execution::seq,
        iterator<std::int64_t>{0}, sentinel<std::int64_t>{100}, &myfunction);

    HPX_TEST_EQ(*iter, std::int64_t(100));

    iter = hpx::ranges::for_each(hpx::execution::par, iterator<std::int64_t>{0},
        sentinel<std::int64_t>{100}, &myfunction);

    HPX_TEST_EQ(*iter, std::int64_t(100));
}

int hpx_main()
{
    test_begin_end_iterator();
    test_invoke_projected();

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ(hpx::local::init(hpx_main, argc, argv), 0);
    return hpx::util::report_errors();
}
