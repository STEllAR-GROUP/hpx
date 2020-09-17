//  Copyright (c) 2018 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx.hpp>
#include <hpx/hpx_main.hpp>

#include <iostream>

HPX_REGISTER_CHANNEL(int)

void f1(hpx::lcos::channel<int>& c1)
{
    for (std::size_t i = 0; true; ++i)
    {
        std::cout << "First Stage: " << i
            << ". Executed on locality " << hpx::get_locality_id()
            << " " << hpx::get_locality_name()
            << '\n';
        c1.set(i);
        hpx::this_thread::sleep_for(std::chrono::seconds(5));
    }
    c1.close();
}

int main()
{
    hpx::lcos::channel<int> c1(hpx::find_here());
    c1.register_as("pipeline/emitter");

    f1(c1);

    return 0;
}
#endif
