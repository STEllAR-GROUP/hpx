////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2020 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

inline constexpr bool test1 = true;

template <typename T>
inline constexpr bool test2 = true;

struct foo
{
};

int main()
{
    static_assert(test1 == test2<foo>, "...");
}
