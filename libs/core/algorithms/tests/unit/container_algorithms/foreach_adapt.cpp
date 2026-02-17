//  Copyright (c) 2020 Giannis Gonidelis
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/algorithm.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>

#include <cstdint>
#include <iterator>
#include <numeric>
#include <vector>

#include "test_utils.hpp"

void myfunction(std::int64_t) {}

template <typename IteratorTag>
void test_invoke_projected(IteratorTag)
{
    using base_iterator = std::vector<std::int64_t>::iterator;
    using iterator = test::test_iterator<base_iterator, IteratorTag>;
    using sentinel = test::sentinel_from_iterator<iterator>;

    std::vector<std::int64_t> c(100);
    std::iota(std::begin(c), std::end(c), 0);

    iterator iter = hpx::ranges::for_each(hpx::execution::seq,
        iterator(std::begin(c)), sentinel(iterator(std::end(c))), myfunction);

    HPX_TEST(iter == iterator(std::end(c)));

    iter = hpx::ranges::for_each(hpx::execution::par, iterator(std::begin(c)),
        sentinel(iterator(std::end(c))), myfunction);

    HPX_TEST(iter == iterator(std::end(c)));
}

template <typename IteratorTag>
void test_begin_end_iterator(IteratorTag)
{
    using base_iterator = std::vector<std::int64_t>::iterator;
    using iterator = test::test_iterator<base_iterator, IteratorTag>;
    using sentinel = test::sentinel_from_iterator<iterator>;

    std::vector<std::int64_t> c(100);
    std::iota(std::begin(c), std::end(c), 0);

    iterator iter = hpx::ranges::for_each(hpx::execution::seq,
        iterator(std::begin(c)), sentinel(iterator(std::end(c))), &myfunction);

    HPX_TEST(iter == iterator(std::end(c)));

    iter = hpx::ranges::for_each(hpx::execution::par, iterator(std::begin(c)),
        sentinel(iterator(std::end(c))), &myfunction);

    HPX_TEST(iter == iterator(std::end(c)));
}

int hpx_main()
{
    test_begin_end_iterator(std::random_access_iterator_tag());
    test_invoke_projected(std::random_access_iterator_tag());

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ(hpx::local::init(hpx_main, argc, argv), 0);
    return hpx::util::report_errors();
}
