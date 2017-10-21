//  Copyright (c) 2017 Jeff Trull
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/parallel/algorithms/merge.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char* argv[])
{
    // By default this should run on all available cores
    std::vector<std::string> const cfg = {
        "hpx.os_threads=1"
    };

    // Initialize and run HPX
    return hpx::init(argc, argv, cfg);
}

int hpx_main(int argc, char **argv)
{
    // these two vectors are sorted by the first value of each tuple
    std::vector<std::tuple<int, char>> a1{
        {1, 'a'},
        {2, 'b'},
        {3, 'a'},
        {3, 'b'},
        {4, 'a'},
        {5, 'a'},
        {5, 'b'}
    };
    std::vector<std::tuple<int, char>> a2{
        {0, 'c'},
        {3, 'c'},
        {4, 'c'},
        {5, 'c'}
    };

    std::vector<std::tuple<int, char>> result(a1.size() + a2.size());
    std::vector<std::tuple<int, char>> solution(a1.size() + a2.size());

    // I expect a stable merge to order {3, 'a'} and {3, 'b'} before {3, 'c'}
    // because they come from the first sequence
    hpx::parallel::merge(
        hpx::parallel::execution::par,
        a1.begin(), a1.end(),
        a2.begin(), a2.end(),
        result.begin(),
        [](auto const& a, auto const& b)
        {
            return std::get<0>(a) < std::get<0>(b);
        });
    std::merge(
        a1.begin(), a1.end(),
        a2.begin(), a2.end(),
        solution.begin(),
        [](auto const& a, auto const& b)
        {
            return std::get<0>(a) < std::get<0>(b);
        });

    bool equality = std::equal(
        result.begin(), result.end(),
        solution.begin(), solution.end());

    HPX_TEST(equality);

    // Expect {3, 'c'}, {3, 'a'}, {3, 'b'} in order.
    hpx::parallel::merge(
        hpx::parallel::execution::par,
        a2.begin(), a2.end(),
        a1.begin(), a1.end(),
        result.begin(),
        [](auto const& a, auto const& b)
        {
            return std::get<0>(a) < std::get<0>(b);
        });
    std::merge(
        a2.begin(), a2.end(),
        a1.begin(), a1.end(),
        solution.begin(),
        [](auto const& a, auto const& b)
        {
            return std::get<0>(a) < std::get<0>(b);
        });
    
    equality = std::equal(
        result.begin(), result.end(),
        solution.begin(), solution.end());

    HPX_TEST(equality);

    return hpx::finalize();
}
