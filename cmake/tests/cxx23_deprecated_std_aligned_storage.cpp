//  Copyright (c) 2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// test for deprecation of std::aligned_storage_t (C++ 23)

int main()
{
#if __cplusplus < 202302L
#error "std::aligned_storage_t is not deprecated before C++23"
#endif
    return 0;
}
