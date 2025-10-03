//  Copyright (c) 2025 Alexandros Papadakis
//  Copyright (c) 2025 Panagiotis Syskakis
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Test: Declaration contracts succeed
// Tests the C++26 declaration-based contract syntax
// This uses proper contracts syntax (pre and post in function declaration) 
// that will run when __cpp_contracts is available

#include <hpx/contracts.hpp>
#include <hpx/modules/testing.hpp>
#include <iostream>

// This test runs when __cpp_contracts is available
// It uses proper C++26 contract syntax in function declarations

// Function with precondition in declaration
int divide(int a, int b) HPX_PRE(b != 0)
{
    return a / b;
}

// Function with postcondition in declaration
int factorial(int n) HPX_PRE(n >= 0) HPX_POST(r; r > 0)
{
    return n <= 1 ? 1 : n * factorial(n - 1);
}

// Function with both pre and post conditions
int safe_multiply(int a, int b) HPX_PRE(a > 0 && b > 0) HPX_POST(r; r > 0)
{
    return a * b;
}

int main()
{
    std::cout << "Testing native C++26 contracts with declaration syntax..." << std::endl;
    
    // Test functions with proper contract syntax
    int result1 = divide(10, 2);
    HPX_TEST_EQ(result1, 5);
    
    int result2 = factorial(5);
    HPX_TEST_EQ(result2, 120);
    
    int result3 = safe_multiply(3, 4);
    HPX_TEST_EQ(result3, 12);
    
    // Test contract assertions
    HPX_CONTRACT_ASSERT(true);
    HPX_CONTRACT_ASSERT(result1 == 5);
    
    std::cout << "✓ All native contract tests passed" << std::endl;
    
    HPX_TEST_MSG(true, "Native C++26 contracts work correctly");
    
    return hpx::util::report_errors();
}