//  Copyright (c) 2020 Giannis Gonidelis
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/iterator_support/traits/is_sentinel_for.hpp>
#include <hpx/testing.hpp>
#include "iter_sent.hpp"

#include <vector>


void is_sentinel_for()
{
    //Iterator<std::int64_t> iter{0};
    //Sentinel<int64_t> sent{100};

    HPX_TEST_MSG((hpx::traits::is_sentinel_for<Iterator<std::int64_t>, Sentinel<int64_t>>::value == true), "sent for iter");
    //HPX_TEST_MSG((hpx::traits::is_sentinel_for<Iterator<std::int64_t>, std::int64_t>::value == true), "sent for iter");
    HPX_TEST_MSG((hpx::traits::is_sentinel_for< std::vector<std::int64_t>, int64_t>::value == true), "sent for iter");


    //static_assert(hpx::traits::is_sentinel_for<Iterator<std::int64_t>,
    //                  Sentinel<int64_t>>::value,
    //    "Nop");
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    {
        is_sentinel_for();
    }

    return hpx::util::report_errors();
}
