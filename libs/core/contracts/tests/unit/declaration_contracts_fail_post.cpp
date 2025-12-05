//  Copyright (c) 2025 Alexandros Papadakis
//  Copyright (c) 2025 Panagiotis Syskakis
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Test: Declaration contracts fail (postcondition)
// Tests C++26 declaration-based postcondition failure
// This uses proper contracts syntax (post in function declaration)
// Expected to fail when __cpp_contracts is available

#include <hpx/contracts.hpp>

// Function with postcondition that requires positive result
int get_positive_number(int) HPX_POST(r : r > 0)
{
    return -10;    // This violates the postcondition r > 0
}

int main()
{
    // This should trigger postcondition violation when __cpp_contracts is
    // available
    [[maybe_unused]] int result = get_positive_number(5);
    return 0;
}
