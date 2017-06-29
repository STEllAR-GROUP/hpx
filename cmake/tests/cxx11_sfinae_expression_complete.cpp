////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2017 Denis Blank
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/util/pack_traversal.hpp>

#include <type_traits>
#include <utility>
#include <vector>

struct all_map_float
{
    template <typename T>
    float operator()(T el) const
    {
        return float(el + 1.f);
    }
};

int main(int, char**)
{
    auto res = hpx::util::map_pack(all_map_float{},
        0,
        1.f,
        hpx::util::make_tuple(1.f, 3),
        std::vector<std::vector<int>>{{1, 2}, {4, 5}},
        std::vector<std::vector<float>>{{1.f, 2.f}, {4.f, 5.f}},
        2);

    auto expected = hpx::util::make_tuple(    // ...
        1.f,
        2.f,
        hpx::util::make_tuple(2.f, 4.f),
        std::vector<std::vector<float>>{{2.f, 3.f}, {5.f, 6.f}},
        std::vector<std::vector<float>>{{2.f, 3.f}, {5.f, 6.f}},
        3.f);

    static_assert(std::is_same<decltype(res), decltype(expected)>::value,
        "Type mismatch!");

    (void) res;
    (void) expected;
}
