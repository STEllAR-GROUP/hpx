//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/modules/testing.hpp>

///////////////////////////////////////////////////////////////////////////////
int main()
{
    hpx::shared_future<int> f1 = hpx::make_ready_future(42);

    hpx::future<int> f2 = f1.then(
        [](hpx::shared_future<int>&&) { return hpx::make_ready_future(43); });

    HPX_TEST_EQ(f1.get(), 42);
    HPX_TEST_EQ(f2.get(), 43);

    return hpx::util::report_errors();
}
