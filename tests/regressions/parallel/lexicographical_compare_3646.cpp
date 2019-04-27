//  Copyright (c) 2019 Jan Melech
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// #3646: Parallel algorithms should accept iterator/sentinel pairs

#include <hpx/hpx_main.hpp>
#include <hpx/include/parallel_lexicographical_compare.hpp>
#include <hpx/util/lightweight_test.hpp>
#include "iter_sent.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <vector>

template<std::int64_t startValue1, std::int64_t stopValue1, 
        std::int64_t startValue2, std::int64_t stopValue2>
void checkCase ()
{
    using Iter1 = Iterator<std::int64_t, stopValue1>;
    using Iter2 = Iterator<std::int64_t, stopValue2>;
    using Sent = Sentinel<std::int64_t>;

    std::vector<int64_t> t1(stopValue1, 0);
    std::iota(t1.begin(), t1.end(), startValue1);
    
    std::vector<int64_t> t2(stopValue2, 0);
    std::iota(t2.begin(), t2.end(), startValue2);

    bool expected = std::lexicographical_compare(
        t1.begin(), t1.end(), t2.begin(), t2.end());

    auto hpxResult = hpx::parallel::lexicographical_compare(
            hpx::parallel::execution::seq,
            Iter1{startValue1}, Sent{}, Iter2{startValue2}, Sent{});

    HPX_TEST_EQ(expected, hpxResult);

    hpxResult = hpx::parallel::lexicographical_compare(
            hpx::parallel::execution::par,
            Iter1{startValue1}, Sent{}, Iter2{startValue2}, Sent{});

    HPX_TEST_EQ(expected, hpxResult);
}

void test_lexicographical_compare()
{
    checkCase<0, 100, 0, 99>();
    checkCase<0, 100, 1, 99>();
    checkCase<1, 100, 0, 99>();

    checkCase<0, 100, 0, 100>();
    checkCase<0, 100, 1, 100>();
    checkCase<1, 100, 0, 100>();
        
    checkCase<0, 100, 0, 101>();
    checkCase<0, 100, 1, 101>();
    checkCase<1, 100, 0, 101>();
}

int main()
{
    test_lexicographical_compare();
    return hpx::util::report_errors();
}
