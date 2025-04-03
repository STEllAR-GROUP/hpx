//  Copyright (c) 2024 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/init.hpp>
#include <hpx/parallel/algorithms/reduce.hpp>
#include <hpx/testing.hpp>

#include <cstddef>
#include <vector>

// A custom proxy type for testing:
// Its implicit conversion to int adds an extra offset (+1)
struct proxy
{
    int value;

    proxy() = default;
    proxy(int v)
      : value(v)
    {
    }

    operator int() const
    {
        return value + 1;
    }
};

// Define addition for proxy objects so that when added as proxy the raw values are summed
proxy operator+(proxy const& a, proxy const& b)
{
    return proxy(a.value + b.value);
}

int hpx_main()
{
    std::vector<proxy> v = {proxy(1), proxy(2), proxy(3)};

    // Test with proxy as input type
    proxy const result1 = hpx::reduce(v.begin(), v.end(), proxy(0));
    HPX_TEST_EQ(result1.value, 6);    // 1 + 2 + 3

    // Test with int as input type (using proxy's implicit conversion)
    int const result2 = hpx::reduce(v.begin(), v.end(), 0);
    HPX_TEST_EQ(result2, 9);    // (1+1) + (2+1) + (3+1)

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ_MSG(hpx::local::init(hpx_main, argc, argv), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}