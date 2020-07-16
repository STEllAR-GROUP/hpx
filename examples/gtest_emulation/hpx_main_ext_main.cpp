//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/modules/testing.hpp>

int report_errors_hpx()
{
    // Check that HPX runtime is started
    HPX_TEST(hpx::threads::get_self_ptr() != nullptr);
    return hpx::util::report_errors();
}
