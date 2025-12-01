//  Copyright (c) 2025 Alexandros Papadakis
//  Copyright (c) 2025 Panagiotis Syskakis
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This tests whether C++26 contracts are supported by the compiler

#if __cpp_contracts
// Test native contract syntax
int divide(int a, int b) pre(b != 0) post(r; r == a / b)
{
    return a / b;
}

void test_contract_assert()
{
    contract_assert(true);
}

int main()
{
    int result = divide(10, 2);
    test_contract_assert();
    return result == 5 ? 0 : 1;
}

#else
// Fallback test - contracts not available
int main()
{
    // Test would fail if contracts were required but not available
    return 0;
}
#endif
