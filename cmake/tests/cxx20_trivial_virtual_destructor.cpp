//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Test if virtual destructors are trivial, so that types with virtual
// destructors can be used in constexpr variables.

struct virtual_destructor
{
    virtual ~virtual_destructor() = default;
};

constexpr virtual_destructor x;

int main() {}
