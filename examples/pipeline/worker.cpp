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

hpx::future<void> f2(hpx::lcos::channel<int>& c1, hpx::lcos::channel<int>& c2)
{
    hpx::future<int> f = c1.get();
    return f.then(
        [&c1, &c2](hpx::future<int> f)
        {
            try
            {
                int i = f.get();
                std::cout << "Second stage: " << i
                    << ". Executed on locality " << hpx::get_locality_id()
                    << " " << hpx::get_locality_name()
                    << '\n';
                c2.set(i + 1);
                hpx::this_thread::sleep_for(std::chrono::microseconds(10));
                return f2(c1, c2);
            }
            catch(...)
            {
                c2.close();
                return hpx::make_ready_future();
            }
        }
    );
}

int main()
{
    std::cout << "Starting worker\n";
    hpx::lcos::channel<int> c1;
    hpx::lcos::channel<int> c2;

    c1.connect_to("pipeline/emitter");
    c2.connect_to("pipeline/collector");

    f2(c1, c2).get();

    return 0;
}
#endif
