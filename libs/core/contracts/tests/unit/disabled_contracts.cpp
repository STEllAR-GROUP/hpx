//  Copyright (c) 2025 Alexandros Papadakis
//  Copyright (c) 2025 Panagiotis Syskakis
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/contracts.hpp>
#include <hpx/modules/testing.hpp>

// This test runs when HPX_WITH_CONTRACTS=OFF
// Verifies that PRE/POST are disabled but CONTRACT_ASSERT still works

int main() 
HPX_PRE(false) 
HPX_POST(false) // Should be no-op, not trigger assertion
{
    // This should still work (maps to HPX_ASSERT)
    HPX_CONTRACT_ASSERT(true);
    
    HPX_TEST(true);  // If we get here, disabled mode works correctly
    
    return hpx::util::report_errors();
}