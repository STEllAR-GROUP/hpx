//  Copyright (c) 2022 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test checks that unwrapping of nullary functions works as expected (see
// #6045)

#include <hpx/modules/pack_traversal.hpp>
#include <hpx/modules/testing.hpp>

bool called = false;

int main()
{
    hpx::unwrapping([]() { called = true; })();
    HPX_TEST(called);
    return hpx::util::report_errors();
}
