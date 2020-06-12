//  Copyright (c) 2020 Giannis Gonidelis
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/iterator_support/traits/is_sentinel_for.hpp>
#include <hpx/testing.hpp>
#include "iter_sent.hpp"

#include <string>
#include <vector>

void is_sentinel_for()
{
    HPX_TEST_MSG((hpx::traits::is_sentinel_for<Sentinel<int64_t>,
                      Iterator<std::int64_t>>::value == true),
        "sent for iter");

    HPX_TEST_MSG(
        (hpx::traits::is_sentinel_for<std::int64_t, std::int64_t>::value ==
            false),
        "incompatible pair int - int");

    HPX_TEST_MSG((hpx::traits::is_sentinel_for<std::vector<int>::iterator,
                      std::vector<int>::iterator>::value == true),
        "iterator begin / end pair");

    HPX_TEST_MSG((hpx::traits::is_sentinel_for<std::string,
                      std::string::iterator>::value == false),
        "string / iterator pair");
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    {
        is_sentinel_for();
    }

    return hpx::util::report_errors();
}
