//  Copyright (c) 2019 Piotr Mikolajczyk
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// #3646: Parallel algorithms should accept iterator/sentinel pairs

#include <hpx/hpx_main.hpp>
#include <hpx/include/parallel_count.hpp>
#include <hpx/modules/testing.hpp>

#include <cstddef>
#include <cstdint>
#include <iterator>

#include "iter_sent.hpp"

struct BitCountingIterator : public Iterator<std::int64_t>
{
    using difference_type = std::ptrdiff_t;
    using value_type = std::int64_t;
    using iterator_category = std::forward_iterator_tag;
    using pointer = std::int64_t const*;
    using reference = std::int64_t const&;

    explicit BitCountingIterator(int64_t initialState)
      : Iterator<int64_t>(initialState)
    {
    }

    std::int64_t operator*() const override
    {
        return countBits(this->state);
    }

    std::int64_t operator[](difference_type n) const override
    {
        return countBits(this->state + n);
    }

    BitCountingIterator& operator++()
    {
        ++(this->state);
        return *this;
    }

    BitCountingIterator operator++(int)
    {
        auto copy = *this;
        ++(*this);
        return copy;
    }

    BitCountingIterator& operator--()
    {
        --(this->state);
        return *this;
    }

    BitCountingIterator operator--(int)
    {
        auto copy = *this;
        --(*this);
        return copy;
    }

private:
    std::int64_t countBits(std::int64_t v) const
    {
        int counter = 0;
        while (v != 0)
        {
            if (v & 1)
                ++counter;
            v >>= 1;
        }
        return counter;
    }
};

void test_count()
{
    using Iter = BitCountingIterator;
    using Sent = Sentinel<std::int64_t>;

    auto stdResult = std::count(Iter{0}, Iter{33}, std::int64_t{1});

    auto result = hpx::ranges::count(
        hpx::parallel::execution::seq, Iter{0}, Sent{33}, std::int64_t{1});

    HPX_TEST_EQ(result, stdResult);

    result = hpx::ranges::count(
        hpx::parallel::execution::par, Iter{0}, Sent{33}, std::int64_t{1});

    HPX_TEST_EQ(result, stdResult);
}

void test_count_if()
{
    using Iter = BitCountingIterator;
    using Sent = Sentinel<std::int64_t>;

    auto predicate = [](std::int64_t v) { return v == 1; };
    auto stdResult = std::count_if(Iter{0}, Iter{33}, predicate);

    Iter::difference_type result = hpx::ranges::count_if(
        hpx::parallel::execution::seq, Iter{0}, Sent{33}, predicate);

    HPX_TEST_EQ(result, stdResult);

    result = hpx::ranges::count_if(
        hpx::parallel::execution::par, Iter{0}, Sent{33}, predicate);

    HPX_TEST_EQ(result, stdResult);
}

int main()
{
    test_count();
    test_count_if();
    return hpx::util::report_errors();
}
