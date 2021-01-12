//  Copyright (c) 2020 Giannis Gonidelis
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/iterator_support/traits/is_sentinel_for.hpp>
#include <hpx/modules/testing.hpp>
#include "iter_sent.hpp"

#include <cstdint>
#include <string>
#include <vector>

void is_sentinel_for()
{
    HPX_TEST_MSG((hpx::traits::is_sentinel_for<sentinel<int64_t>,
                      iterator<std::int64_t>>::value == true),
        "Sentinel value is not proper for given iterator");

    HPX_TEST_MSG(
        (hpx::traits::is_sentinel_for<std::int64_t, std::int64_t>::value ==
            false),
        "Integer - integer pair is ncompatible pair");

    HPX_TEST_MSG((hpx::traits::is_sentinel_for<std::vector<int>::iterator,
                      std::vector<int>::iterator>::value == true),
        "Incompatible begin - end iterator pair on vector");

    HPX_TEST_MSG((hpx::traits::is_sentinel_for<std::string,
                      std::string::iterator>::value == false),
        "String - string::iterator is ncompatible pair");
}

///////////////////////////////////////////////////////////////////////////////
int main()
{
    {
        is_sentinel_for();
    }

    return hpx::util::report_errors();
}
