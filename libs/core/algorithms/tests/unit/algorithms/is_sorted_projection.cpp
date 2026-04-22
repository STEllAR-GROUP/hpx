//  Copyright (c) 2024-2026 STE||AR-Group
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Regression test: hpx::is_sorted, hpx::is_sorted_until, and
// hpx::is_partitioned did not expose a projection parameter in their CPO
// (tag_fallback_invoke) overloads, despite the internal algorithm
// implementations fully supporting projections. This file validates that
// all three algorithms accept a projection with seq and par policies.

#include <hpx/algorithm.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>

#include <cstddef>
#include <functional>
#include <utility>
#include <vector>

// ---------------------------------------------------------------------------
// is_sorted with projection
// ---------------------------------------------------------------------------
void test_is_sorted_projection()
{
    using namespace hpx::execution;
    using element = std::pair<int, int>;

    // sorted by .second: 1, 3, 5, 8
    std::vector<element> sorted_c = {{10, 1}, {20, 3}, {30, 5}, {40, 8}};
    // NOT sorted by .second: 1, 5, 3, 8
    std::vector<element> unsorted_c = {{10, 1}, {20, 5}, {30, 3}, {40, 8}};

    auto proj = [](element const& p) { return p.second; };

    // sequential, no policy
    HPX_TEST(hpx::is_sorted(
        sorted_c.begin(), sorted_c.end(), std::less<int>{}, proj));
    HPX_TEST(!hpx::is_sorted(
        unsorted_c.begin(), unsorted_c.end(), std::less<int>{}, proj));

    // seq policy
    HPX_TEST(hpx::is_sorted(
        seq, sorted_c.begin(), sorted_c.end(), std::less<int>{}, proj));
    HPX_TEST(!hpx::is_sorted(
        seq, unsorted_c.begin(), unsorted_c.end(), std::less<int>{}, proj));

    // par policy
    HPX_TEST(hpx::is_sorted(
        par, sorted_c.begin(), sorted_c.end(), std::less<int>{}, proj));
    HPX_TEST(!hpx::is_sorted(
        par, unsorted_c.begin(), unsorted_c.end(), std::less<int>{}, proj));

    // par(task) policy
    HPX_TEST(hpx::is_sorted(
        par(task), sorted_c.begin(), sorted_c.end(), std::less<int>{}, proj)
            .get());
    HPX_TEST(!hpx::is_sorted(
        par(task), unsorted_c.begin(), unsorted_c.end(), std::less<int>{}, proj)
            .get());

    // empty range always returns true
    HPX_TEST(hpx::is_sorted(
        par, sorted_c.begin(), sorted_c.begin(), std::less<int>{}, proj));
    // single element range always returns true
    HPX_TEST(hpx::is_sorted(
        par, sorted_c.begin(), sorted_c.begin() + 1, std::less<int>{}, proj));
}

// ---------------------------------------------------------------------------
// is_sorted_until with projection
// ---------------------------------------------------------------------------
void test_is_sorted_until_projection()
{
    using namespace hpx::execution;
    using element = std::pair<int, int>;

    // sorted by .second: 1, 3, 8; then unsorted at index 3 (value 5)
    std::vector<element> c = {{10, 1}, {20, 3}, {30, 8}, {40, 5}, {50, 9}};

    auto proj = [](element const& p) { return p.second; };

    // sequential, no policy
    auto it_seq =
        hpx::is_sorted_until(c.begin(), c.end(), std::less<int>{}, proj);
    HPX_TEST(it_seq != c.end());
    HPX_TEST_EQ(it_seq->second, 5);

    // seq policy
    auto it_seq2 =
        hpx::is_sorted_until(seq, c.begin(), c.end(), std::less<int>{}, proj);
    HPX_TEST(it_seq2 != c.end());
    HPX_TEST_EQ(it_seq2->second, 5);

    // par policy
    auto it_par =
        hpx::is_sorted_until(par, c.begin(), c.end(), std::less<int>{}, proj);
    HPX_TEST(it_par != c.end());
    HPX_TEST_EQ(it_par->second, 5);

    // par(task) policy
    auto it_task = hpx::is_sorted_until(
        par(task), c.begin(), c.end(), std::less<int>{}, proj)
                       .get();
    HPX_TEST(it_task != c.end());
    HPX_TEST_EQ(it_task->second, 5);

    // fully sorted range should return end
    std::vector<element> fully_sorted = {{10, 1}, {20, 3}, {30, 5}};
    auto it_end = hpx::is_sorted_until(
        par, fully_sorted.begin(), fully_sorted.end(), std::less<int>{}, proj);
    HPX_TEST(it_end == fully_sorted.end());

    // empty range should return end (begin == end)
    auto it_empty = hpx::is_sorted_until(par, fully_sorted.begin(),
        fully_sorted.begin(), std::less<int>{}, proj);
    HPX_TEST(it_empty == fully_sorted.begin());
}

// ---------------------------------------------------------------------------
// is_partitioned with projection
// ---------------------------------------------------------------------------
void test_is_partitioned_projection()
{
    using namespace hpx::execution;
    using element = std::pair<int, int>;

    // partitioned by .second % 2 == 0: all even seconds first, then odd
    // second values: 2, 4, 1, 3 => partitioned (even before odd)
    std::vector<element> partitioned_c = {{10, 2}, {20, 4}, {30, 1}, {40, 3}};
    // NOT partitioned by .second % 2 == 0: 2, 1, 4, 3
    std::vector<element> not_partitioned_c = {
        {10, 2}, {20, 1}, {30, 4}, {40, 3}};

    auto proj = [](element const& p) { return p.second; };
    auto pred = [](int v) { return v % 2 == 0; };

    // sequential, no policy
    HPX_TEST(hpx::is_partitioned(
        partitioned_c.begin(), partitioned_c.end(), pred, proj));
    HPX_TEST(!hpx::is_partitioned(
        not_partitioned_c.begin(), not_partitioned_c.end(), pred, proj));

    // seq policy
    HPX_TEST(hpx::is_partitioned(
        seq, partitioned_c.begin(), partitioned_c.end(), pred, proj));
    HPX_TEST(!hpx::is_partitioned(
        seq, not_partitioned_c.begin(), not_partitioned_c.end(), pred, proj));

    // par policy
    HPX_TEST(hpx::is_partitioned(
        par, partitioned_c.begin(), partitioned_c.end(), pred, proj));
    HPX_TEST(!hpx::is_partitioned(
        par, not_partitioned_c.begin(), not_partitioned_c.end(), pred, proj));

    // par(task) policy
    HPX_TEST(hpx::is_partitioned(
        par(task), partitioned_c.begin(), partitioned_c.end(), pred, proj)
            .get());
    HPX_TEST(!hpx::is_partitioned(par(task), not_partitioned_c.begin(),
        not_partitioned_c.end(), pred, proj)
            .get());

    // empty range is always partitioned
    HPX_TEST(hpx::is_partitioned(
        par, partitioned_c.begin(), partitioned_c.begin(), pred, proj));
    // single element range is always partitioned
    HPX_TEST(hpx::is_partitioned(
        par, partitioned_c.begin(), partitioned_c.begin() + 1, pred, proj));
}

int hpx_main(hpx::program_options::variables_map&)
{
    test_is_sorted_projection();
    test_is_sorted_until_projection();
    test_is_partitioned_projection();

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    std::vector<std::string> const cfg = {"hpx.os_threads=all"};

    hpx::local::init_params init_args;
    init_args.cfg = cfg;

    HPX_TEST_EQ_MSG(hpx::local::init(hpx_main, argc, argv, init_args), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
