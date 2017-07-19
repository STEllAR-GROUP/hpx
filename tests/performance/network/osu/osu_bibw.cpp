//  Copyright (c) 2013-2015 Hartmut Kaiser
//  Copyright (c) 2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Bidirectional network bandwidth test

#include <hpx/hpx.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/include/serialization.hpp>
#include <hpx/include/parallel_for_each.hpp>

#include <boost/range/irange.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
#define LOOP_SMALL  100
#define WINDOW_SIZE_SMALL  2048
#define SKIP_SMALL  10

#define LOOP_LARGE  20
#define WINDOW_SIZE_LARGE  2048
#define SKIP_LARGE  2

#define LARGE_MESSAGE_SIZE  8192

#define MAX_MSG_SIZE (1<<22)
#define MAX_ALIGNMENT 65536
#define SEND_BUFSIZE (MAX_MSG_SIZE + MAX_ALIGNMENT)

char send_buffer[SEND_BUFSIZE];

///////////////////////////////////////////////////////////////////////////////
void isend(hpx::serialization::serialize_buffer<char> const& receive_buffer) {}
HPX_PLAIN_ACTION(isend);

///////////////////////////////////////////////////////////////////////////////
hpx::future<void>
send_async(hpx::naming::id_type dest, std::size_t size, std::size_t window_size)
{
    typedef hpx::serialization::serialize_buffer<char> buffer_type;

    using hpx::parallel::for_each;
    using hpx::parallel::execution::par;
    using hpx::parallel::execution::task;

    std::size_t const start = 0;

    auto range = boost::irange(start, window_size);
    return for_each(
        par(task), std::begin(range), std::end(range),
        [=](std::uint64_t j)
        {
            // Note: The original benchmark uses MPI_Isend which does not
            //       create a copy of the passed buffer.
            isend_action send;
            send(dest, buffer_type(send_buffer, size, buffer_type::reference));
        });
}

///////////////////////////////////////////////////////////////////////////////
hpx::serialization::serialize_buffer<char> irecv(std::size_t size)
{
    typedef hpx::serialization::serialize_buffer<char> buffer_type;
    return buffer_type(send_buffer, size, buffer_type::reference);
}
HPX_PLAIN_ACTION(irecv);

///////////////////////////////////////////////////////////////////////////////
hpx::future<void>
recv_async(hpx::naming::id_type dest, std::size_t size, std::size_t window_size)
{
    typedef hpx::serialization::serialize_buffer<char> buffer_type;

    using hpx::parallel::for_each;
    using hpx::parallel::execution::par;
    using hpx::parallel::execution::task;

    std::size_t const start = 0;

    auto range = boost::irange(start, window_size);
    return for_each(
        par(task), std::begin(range), std::end(range),
        [=](std::uint64_t j)
        {
            irecv_action recv;
            recv(dest, size);
        });
}

///////////////////////////////////////////////////////////////////////////////
void print_header()
{
    hpx::cout << "# OSU HPX Bi-Directional Test\n"
              << "# Size    Bandwidth (MB/s)"
              << std::endl;
}

///////////////////////////////////////////////////////////////////////////////
void run_benchmark(boost::program_options::variables_map & vm)
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

        if (size > LARGE_MESSAGE_SIZE) {
            loop = LOOP_LARGE;
            skip = SKIP_LARGE;
            window_size = WINDOW_SIZE_LARGE;
        }

        hpx::util::high_resolution_timer t;

        for(std::size_t i = 0; i < loop + skip; ++i)
        {
            if (i == skip)      // don't measure during warm-up phase
                t.restart();

            hpx::future<void> recv =
                hpx::async(recv_async, there, size, window_size);

            send_async(there, size, window_size);

            recv.wait();
        }

        double bw = (size / 1e6 * loop * window_size * 2) / t.elapsed();

        hpx::cout << std::left << std::setw(10) << size << bw << std::endl;
    }
}
