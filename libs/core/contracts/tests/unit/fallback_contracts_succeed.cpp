//  Copyright (c) 2025 Alexandros Papadakis
//  Copyright (c) 2025 Panagiotis Syskakis
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Test: Simple contracts succeed
// Tests the simple contract syntax that works in both native and fallback modes
// HPX_PRE(condition), HPX_POST(condition), HPX_CONTRACT_ASSERT(condition)

#include <hpx/contracts.hpp>

int main()
{
    HPX_CONTRACT_ASSERT(true);
    HPX_PRE(true);
    HPX_POST(true);
}