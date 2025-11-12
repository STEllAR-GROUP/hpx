//  Copyright (c) 2025 Alexandros Papadakis
//  Copyright (c) 2025 Panagiotis Syskakis
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Test: Declaration contracts fail (precondition)
// Tests C++26 declaration-based precondition failure
// This uses proper contracts syntax (pre in function declaration)
// Expected to fail when __cpp_contracts is available

#include <hpx/contracts.hpp>

// Function with precondition that requires positive input
int multiply_positive(int const x) HPX_PRE(x > 0)
{
    return x * 2;
}

int main()
{
    // This should trigger precondition violation when __cpp_contracts is available
    return multiply_positive(-5);    // Violates x > 0
}
