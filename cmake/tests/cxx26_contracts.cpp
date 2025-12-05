//  Copyright (c) 2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// test for availability of C++26 contracts

#if !defined(__cpp_contracts)
#error "__cpp_contracts not defined, assume contracts are not supported"
#endif

#include <contracts>

// Test actual contract syntax support (for experimental implementations)
int main() pre(true) post(r : r == 0)
{
    contract_assert(true);
    return 0;
}
