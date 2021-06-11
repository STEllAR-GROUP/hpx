//  Copyright (c) 2021 Srinivas Yadav
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// test for availability of std experimental simd (C++20)

// Enable this test only for GCC Compilers as simd header is not
// completely implemented for other compilers.
#if defined(__GNUC__) && !defined(__clang__)
#include <experimental/simd>
#endif

int main()
{
    std::experimental::simd<int> s = std::experimental::simd<int>();
    (void) s;
    return 0;
}
