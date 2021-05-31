//  Copyright (c) 2021 Srinivas Yadav
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// test for availability of std experimental simd (C++20)

#include <experimental/simd>

int main()
{
    std::experimental::simd<int> s = std::experimental::simd<int>();
    (void) s;
    return 0;
}
