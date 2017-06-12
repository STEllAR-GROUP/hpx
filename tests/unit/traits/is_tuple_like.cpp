//  Copyright (c) 2017 Denis Blank
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <utility>
#include <vector>

#if defined(HPX_HAVE_CXX11_STD_ARRAY)
#include <array>
#endif

#include <hpx/config.hpp>
#include <hpx/traits/is_tuple_like.hpp>
#include <hpx/util/lightweight_test.hpp>
#include <hpx/util/tuple.hpp>

void tuple_like_true()
{
    using hpx::traits::is_tuple_like;

    HPX_TEST_EQ((is_tuple_like<hpx::util::tuple<int, int, int>>::value), true);
    HPX_TEST_EQ((is_tuple_like<std::pair<int, int>>::value), true);

#if defined(HPX_HAVE_CXX11_STD_ARRAY)
    HPX_TEST_EQ((is_tuple_like<std::array<int, 4>>::value), true);
#endif
}

void tuple_like_false()
{
    using hpx::traits::is_tuple_like;

    HPX_TEST_EQ((is_tuple_like<int>::value), false);
    HPX_TEST_EQ((is_tuple_like<std::vector<int>>::value), false);
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    {
        tuple_like_true();
        tuple_like_false();
    }

    return hpx::util::report_errors();
}
