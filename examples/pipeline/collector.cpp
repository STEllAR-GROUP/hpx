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

hpx::future<void> f3(hpx::lcos::channel<int>& c2)
{
    hpx::future<int> f = c2.get();
    return f.then(
        [&c2](hpx::future<int> f)
        {
            try
            {
                std::cout << "Final stage: " << f.get()
                    << ". Executed on locality " << hpx::get_locality_id()
                    << " " << hpx::get_locality_name()
                    << '\n';
                return f3(c2);
            }
            catch(...)
            {
                std::cout << "Pipeline done.\n";
                return hpx::make_ready_future();
            }
        }
    );
}

int main()
{
    std::cout << "Starting collector\n";
    hpx::lcos::channel<int> c2(hpx::find_here());
    c2.register_as("pipeline/collector");

    f3(c2).get();

    return 0;
}
#endif
