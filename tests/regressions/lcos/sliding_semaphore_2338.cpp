//  Copyright (c) 2016 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// verify #2338 is fixed (Possible race in sliding semaphore)

#include <hpx/hpx_main.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <cstddef>
#include <vector>

// ----------------------------------------------------------------------------
double message_double(double d)
{
    return d;
}
HPX_PLAIN_ACTION(message_double);

// ----------------------------------------------------------------------------
int main()
{
    // use the first remote locality to bounce messages, if possible
    hpx::id_type here = hpx::find_here();

    hpx::id_type there = here;
    std::vector<hpx::id_type> localities = hpx::find_remote_localities();
    if (!localities.empty())
        there = localities[0];

    std::size_t parcel_count = 0;
    std::size_t loop         = 10000;
    std::size_t window_size  = 1;
    std::size_t skip         = 50;

    std::size_t signal_count(0);

    hpx::lcos::local::sliding_semaphore sem(window_size, 0);
    message_double_action msg;

    for (std::size_t i = 0; i < (loop*window_size) + skip; ++i)
    {
        // launch a message to the remote node
        hpx::async(msg, there, 3.5).then(
            hpx::launch::sync,
            // when the message completes, increment our semaphore count
            // so that N are always in flight
            [&,parcel_count](hpx::future<double> &&f) -> void
            {
                sem.signal(parcel_count);
                ++signal_count;
            }
        );

        //
        ++parcel_count;

        //
        sem.wait(parcel_count);
    }

    sem.wait(parcel_count + window_size - 1);

    HPX_TEST_EQ(signal_count, (loop*window_size) + skip);

    return hpx::util::report_errors();
}
