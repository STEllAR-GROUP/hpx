//  Copyright (c) 2017 Igor Krivenko
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/modules/testing.hpp>

#include <cstddef>

typedef hpx::naming::id_type locality_id_t;
HPX_REGISTER_CHANNEL(locality_id_t);

std::atomic<std::size_t> count(0);

int hpx_main()
{
    // List of currently available resources
    hpx::lcos::channel<locality_id_t> free_resources(hpx::find_here());

    std::size_t os_thread_count = hpx::get_os_thread_count();

    // At the beginning all threads on all localities are free
    for (locality_id_t id : hpx::find_all_localities())
    {
        for (std::size_t i = 0; i != os_thread_count; ++i)
        {
            free_resources.set(id);
            ++count;
        }
    }

    for (int i = 0; i < 1000; ++i)
    {
        // Ask for resources
        hpx::shared_future<locality_id_t> target = free_resources.get();

        // Do some work, once we have acquired resources
        hpx::shared_future<int> result = target.then(
            [](hpx::shared_future<locality_id_t>) -> hpx::shared_future<int> {
                --count;
                return hpx::make_ready_future(0);
            });

        // Free resources
        result.then([free_resources, target](hpx::shared_future<int>) mutable {
            ++count;
            free_resources.set(target.get());
        });

        result.get();
    }

    // There might be at least one thread less waiting than are currently
    // pending as the first thread exiting the loop did not request an item
    // off the channel anymore.
    std::size_t remaining_count = free_resources.close(true);
    HPX_TEST_LTE(remaining_count, count.load());

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ(0, hpx::init(argc, argv));
    return hpx::util::report_errors();
}
#endif
