////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2020 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <tuple>
#include <type_traits>

template <typename... Ts>
void test()
{
    static_assert(std::is_same_v<__type_pack_element<0, Ts...>,
        typename std::tuple_element<0, std::tuple<Ts...>>::type>);
    static_assert(std::is_same_v<__type_pack_element<1, Ts...>,
        typename std::tuple_element<1, std::tuple<Ts...>>::type>);
    static_assert(std::is_same_v<__type_pack_element<2, Ts...>,
        typename std::tuple_element<2, std::tuple<Ts...>>::type>);
}

int main()
{
    test<int, float, double>();
}
