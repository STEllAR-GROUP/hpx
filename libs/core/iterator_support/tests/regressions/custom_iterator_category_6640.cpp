//  Copyright (c) 2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/iterator_support/tests/iterator_tests.hpp>
#include <hpx/modules/iterator_support.hpp>
#include <hpx/modules/testing.hpp>

#include <cstdint>
#include <iterator>

// Try to instantiate iterator_facade with a custom iterator_tag
struct test_iterator_tag : std::random_access_iterator_tag
{
};

class test_iterator
  : public hpx::util::iterator_facade<test_iterator, std::int64_t const,
        test_iterator_tag>
{
public:
    test_iterator() = default;

    void increment()
    {
        ++val;
    }

    void decrement()
    {
        --val;
    }

    void advance(std::int64_t const n)
    {
        val += n;
    }

    [[nodiscard]] std::int64_t const& dereference() const
    {
        return val;
    }

    [[nodiscard]] bool equal(test_iterator const& rhs) const
    {
        return val == rhs.val;
    }

    [[nodiscard]] std::int64_t distance_to(test_iterator const& rhs) const
    {
        return rhs.val - val;
    }

    std::int64_t val = 42;
};

int main()
{
    std::int64_t array[] = {42, 43, 44, 45, 46, 47};
    constexpr std::int64_t N = sizeof(array) / sizeof(std::int64_t);

    tests::random_access_iterator_test(test_iterator(), N, array);
    tests::random_access_readable_iterator_test(test_iterator(), N, array);

    return hpx::util::report_errors();
}
