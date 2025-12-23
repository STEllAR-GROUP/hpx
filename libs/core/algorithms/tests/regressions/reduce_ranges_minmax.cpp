//  Copyright (c) 2025 Arivoli Ramamoorthy
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
//  Tests that hpx::ranges::reduce correctly handles non-trivial identity values
//  and accumulator types different from the element type (regression test for
//  incorrect seeding from *part_begin in parallel reduce).

#include <hpx/algorithm.hpp>
#include <hpx/init.hpp>
#include <hpx/iterator_support/tests/iter_sent.hpp>
#include <hpx/modules/testing.hpp>

#include <cstdint>
#include <limits>
#include <utility>

struct minmax_op
{
    using result_type = std::pair<std::int64_t, std::int64_t>;

    result_type operator()(result_type acc, std::int64_t x) const
    {
        return {(std::min) (acc.first, x), (std::max) (acc.second, x)};
    }

    result_type operator()(result_type a, result_type b) const
    {
        return {(std::min) (a.first, b.first), (std::max) (a.second, b.second)};
    }
};

int hpx_main()
{
    using result_type = std::pair<std::int64_t, std::int64_t>;

    const result_type identity{(std::numeric_limits<std::int64_t>::max)(),
        (std::numeric_limits<std::int64_t>::min)()};

    auto result =
        hpx::ranges::reduce(hpx::execution::par, iterator<std::int64_t>{0},
            sentinel<std::int64_t>{100}, identity, minmax_op{});

    HPX_TEST_EQ(result.first, 0);
    HPX_TEST_EQ(result.second, 99);

    auto empty_result =
        hpx::ranges::reduce(hpx::execution::par, iterator<std::int64_t>{0},
            sentinel<std::int64_t>{0}, identity, minmax_op{});

    HPX_TEST_EQ(empty_result.first, identity.first);
    HPX_TEST_EQ(empty_result.second, identity.second);

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ_MSG(hpx::local::init(hpx_main, argc, argv), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
