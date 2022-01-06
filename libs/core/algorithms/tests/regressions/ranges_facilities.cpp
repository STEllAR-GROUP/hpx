//  Copyright (c) 2021 Giannis Gonidelis
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/iterator_support/tests/iter_sent.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/parallel/util/ranges_facilities.hpp>

#include <cstdint>
#include <vector>

void test_ranges_next()
{
    std::vector<std::int16_t> v = {1, 5, 3, 6};
    auto it = v.begin();

    auto next1 = hpx::ranges::next(it);
    HPX_TEST_EQ(*next1, 5);

    auto next2 = hpx::ranges::next(it, 2);
    HPX_TEST_EQ(*next2, 3);

    auto next3 = hpx::ranges::next(it, sentinel<std::int16_t>(3));
    HPX_TEST_EQ(*next3, 3);

    auto next4 = hpx::ranges::next(it, 2, v.end());
    HPX_TEST_EQ(*next4, 3);

    auto next5 = hpx::ranges::next(it, 42, v.end());
    HPX_TEST(next5 == v.end());
}

int main()
{
    test_ranges_next();
    return hpx::util::report_errors();
}
