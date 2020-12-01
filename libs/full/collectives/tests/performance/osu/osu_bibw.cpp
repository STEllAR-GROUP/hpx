//  Copyright (c) 2013-2019 Hartmut Kaiser
//  Copyright (c) 2013 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Bidirectional network bandwidth test

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/algorithm.hpp>
#include <hpx/execution.hpp>
#include <hpx/hpx.hpp>
#include <hpx/serialization.hpp>

#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
#define LOOP_SMALL 100
#define SKIP_SMALL 10

#define LOOP_LARGE 20
#define SKIP_LARGE 2

#if defined(HPX_DEBUG)
#define WINDOW_SIZE_SMALL 32
#define WINDOW_SIZE_LARGE 32
#else
#define WINDOW_SIZE_SMALL 2048
#define WINDOW_SIZE_LARGE 2048
#endif

#define LARGE_MESSAGE_SIZE 8192

#define MAX_MSG_SIZE (1 << 22)
#define MAX_ALIGNMENT 65536
#define SEND_BUFSIZE (MAX_MSG_SIZE + MAX_ALIGNMENT)

char send_buffer[SEND_BUFSIZE];

///////////////////////////////////////////////////////////////////////////////
void isend(hpx::serialization::serialize_buffer<char> const&) {}
HPX_PLAIN_DIRECT_ACTION(isend);

///////////////////////////////////////////////////////////////////////////////
hpx::future<void> send_async(
    hpx::id_type dest, std::size_t size, std::size_t window_size)
{
    using buffer_type = hpx::serialization::serialize_buffer<char>;

    using hpx::for_loop;
    using hpx::execution::par;
    using hpx::execution::task;

    return for_loop(par(task), 0, window_size, [dest, size](std::uint64_t) {
        // Note: The original benchmark uses MPI_Isend which does not
        //       create a copy of the passed buffer.
        isend_action send;
        send(dest, buffer_type(send_buffer, size, buffer_type::reference));
    });
}

///////////////////////////////////////////////////////////////////////////////
hpx::serialization::serialize_buffer<char> irecv(std::size_t size)
{
    using buffer_type = hpx::serialization::serialize_buffer<char>;
    return buffer_type(send_buffer, size, buffer_type::reference);
}
HPX_PLAIN_DIRECT_ACTION(irecv);

///////////////////////////////////////////////////////////////////////////////
void recv_async(hpx::id_type dest, std::size_t size, std::size_t window_size)
{
    using hpx::for_loop;
    using hpx::execution::par;

    for_loop(par, 0, window_size, [dest, size](std::uint64_t) {
        irecv_action recv;
        recv(dest, size);
    });
}

///////////////////////////////////////////////////////////////////////////////
void print_header()
{
    std::cout << "# OSU HPX Bi-Directional Test\n"
              << "# Size    Bandwidth (MB/s)" << std::endl;
}

///////////////////////////////////////////////////////////////////////////////
void run_benchmark(hpx::program_options::variables_map&)
{
    // use the first remote locality to bounce messages, if possible
    hpx::id_type here = hpx::find_here();

    hpx::id_type there = here;
    std::vector<hpx::id_type> localities = hpx::find_remote_localities();
    if (!localities.empty())
        there = localities[0];

    // perform actual measurements
    for (std::size_t size = 1; size <= MAX_MSG_SIZE; size *= 2)
    {
        std::size_t loop = LOOP_SMALL;
        std::size_t skip = SKIP_SMALL;
        std::size_t window_size = WINDOW_SIZE_SMALL;

        if (size > LARGE_MESSAGE_SIZE)
        {
            loop = LOOP_LARGE;
            skip = SKIP_LARGE;
            window_size = WINDOW_SIZE_LARGE;
        }

        hpx::chrono::high_resolution_timer t;

        for (std::size_t i = 0; i < loop + skip; ++i)
        {
            if (i == skip)    // don't measure during warm-up phase
                t.restart();

            hpx::future<void> recv =
                hpx::async(recv_async, there, size, window_size);

            send_async(there, size, window_size);

            recv.wait();
        }

        double bw = (size / 1e6 * loop * window_size * 2) / t.elapsed();

        std::cout << std::left << std::setw(10) << size << bw << std::endl;
    }
}
#endif
