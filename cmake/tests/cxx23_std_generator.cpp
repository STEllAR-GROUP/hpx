//  Copyright (c) 2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// test for availability of std::generator (C++ 23)

#include <generator>
#include <string>

// Test basic generator functionality
std::generator<std::string> test_generator()
{
    co_yield "test";
    co_yield "string";
}

// Test generator with reference type
std::generator<std::string&> test_ref_generator(std::string& str)
{
    co_yield str;
}

int main()
{
    // Test basic generator
    for (const auto& str : test_generator())
    {
        (void)str;
    }

    // Test generator with reference
    std::string test_str = "test";
    for (auto& str : test_ref_generator(test_str))
    {
        (void)str;
    }

    return 0;
}
