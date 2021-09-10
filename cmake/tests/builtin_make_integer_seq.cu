////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2020 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <cstddef>
#include <type_traits>

template <typename T, T... Vs>
struct pack_c
{
};

template <typename = void>
void test()
{
    static_assert(std::is_same<__make_integer_seq<pack_c, std::size_t, 0>,
                      pack_c<std::size_t>>::value,
        "");
    static_assert(std::is_same<__make_integer_seq<pack_c, std::size_t, 1>,
                      pack_c<std::size_t, 0>>::value,
        "");
    static_assert(std::is_same<__make_integer_seq<pack_c, std::size_t, 2>,
                      pack_c<std::size_t, 0, 1>>::value,
        "");
}

int main()
{
    test();
}
