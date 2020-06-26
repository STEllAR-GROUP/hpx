//  Copyright (c) 2017 KADichev
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/modules/collectives.hpp>
#include <hpx/modules/testing.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <vector>

std::size_t const N = 1024;
std::vector<double> buffer;

void vector_bcast(std::vector<double> bcast)
{
    buffer = bcast;
}

HPX_PLAIN_ACTION(vector_bcast);

HPX_REGISTER_BROADCAST_ACTION_DECLARATION(vector_bcast_action)
HPX_REGISTER_BROADCAST_ACTION(vector_bcast_action)

int hpx_main()
{
    std::vector<hpx::id_type> localities = hpx::find_all_localities();
    std::vector<double> bcast_array(N);
    for (std::size_t i = 0; i != N; ++i)
        bcast_array[i] = 3.14;

    for (std::size_t i = 0; i != 5; ++i)
    {
        buffer.clear();

        auto f =
            hpx::lcos::broadcast<vector_bcast_action>(localities, bcast_array);
        hpx::lcos::future_status status = hpx::lcos::future_status::timeout;

        // This test should usually test against future_status == ready. However,
        // under some circumstances, the operation might still take a little
        // longer...
        while (status != hpx::lcos::future_status::ready)
        {
            status = f.wait_for(std::chrono::milliseconds(1000));
            HPX_TEST(status == hpx::lcos::future_status::ready ||
                status == hpx::lcos::future_status::timeout);
        }

        HPX_TEST_EQ(buffer.size(), N);
        HPX_TEST_EQ(buffer.size(), bcast_array.size());

        for (std::size_t i = 0; i != N; ++i)
        {
            HPX_TEST_EQ(buffer[i], bcast_array[i]);
        }
    }

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ(hpx::init(argc, argv), 0);
    return hpx::util::report_errors();
}
