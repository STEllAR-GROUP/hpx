//  Copyright (c) 2017 Denis Blank
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <array>
#include <utility>
#include <vector>

#include <hpx/config.hpp>
#include <hpx/datastructures/traits/is_tuple_like.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/modules/testing.hpp>

void tuple_like_true()
{
    using hpx::traits::is_tuple_like;

    HPX_TEST_EQ((is_tuple_like<hpx::tuple<int, int, int>>::value), true);
    HPX_TEST_EQ((is_tuple_like<std::pair<int, int>>::value), true);
    HPX_TEST_EQ((is_tuple_like<std::array<int, 4>>::value), true);
}

void tuple_like_false()
{
    using hpx::traits::is_tuple_like;

    HPX_TEST_EQ((is_tuple_like<int>::value), false);
    HPX_TEST_EQ((is_tuple_like<std::vector<int>>::value), false);
}

///////////////////////////////////////////////////////////////////////////////
int main()
{
    {
        tuple_like_true();
        tuple_like_false();
    }

    return hpx::util::report_errors();
}
