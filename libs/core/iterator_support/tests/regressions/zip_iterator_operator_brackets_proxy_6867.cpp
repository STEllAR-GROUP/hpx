//  Copyright (c) 2026 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// zip_iterator's operator[] doesn't agree with gcc 16's new standard library
// implementation of `std::sort` (as reported by #6867).

#include <hpx/hpx_main.hpp>
#include <hpx/modules/datastructures.hpp>
#include <hpx/modules/iterator_support.hpp>
#include <hpx/modules/testing.hpp>

#include <utility>
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

    void test_zip_iterator_operator_brackets_proxy()
    {
        std::vector<char> r1 = {'a', 'b'};
        std::vector<double> r2 = {1.0, 2.0};

        auto r = zip(r1, r2);
        auto it = r.begin();

        using std::swap;
        swap(it[0], it[1]);

        HPX_TEST_EQ(r1[0], 'b');
        HPX_TEST_EQ(r1[1], 'a');
        HPX_TEST_EQ(r2[0], 2.0);
        HPX_TEST_EQ(r2[1], 1.0);
    }
}    // namespace

int main()
{
    test_zip_iterator_operator_brackets_proxy();
    return hpx::util::report_errors();
}
