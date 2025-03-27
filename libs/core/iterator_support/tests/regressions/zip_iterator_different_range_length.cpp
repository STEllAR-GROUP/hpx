//  Copyright (c) 2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/modules/datastructures.hpp>
#include <hpx/modules/iterator_support.hpp>
#include <hpx/modules/testing.hpp>

#include <vector>

namespace {

    template <typename R1, typename R2, typename... Rs>
    auto zip(R1&& r1, R2&& r2, Rs&&... rs)
    {
        using namespace hpx::util;
        return iterator_range(
            zip_iterator(r1.begin(), r2.begin(), rs.begin()...),
            zip_iterator(r1.end(), r2.end(), rs.end()...));
    }
}    // namespace

int main()
{
    std::vector<char> r1 = {'a', 'b', 'c', 'd'};
    std::vector<double> r2 = {1.0, 2.0};

    auto const r = zip(r1, r2);

    auto it = r.begin();
    auto const end = r.end();

    HPX_TEST(it != end);
    {
        auto t = *it;
        HPX_TEST_EQ(hpx::get<0>(t), 'a');
        HPX_TEST_EQ(hpx::get<1>(t), 1.0);
    }

    HPX_TEST(++it != end);
    {
        auto t = *it;
        HPX_TEST_EQ(hpx::get<0>(t), 'b');
        HPX_TEST_EQ(hpx::get<1>(t), 2.0);
    }

    HPX_TEST(++it == end);

    return hpx::util::report_errors();
}
