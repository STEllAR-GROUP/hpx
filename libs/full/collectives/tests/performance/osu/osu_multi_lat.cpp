//  Copyright (c) 2013-2025 Hartmut Kaiser
//  Copyright (c) 2013 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Multi latency network test

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/algorithm.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/serialization.hpp>
#include <hpx/include/util.hpp>
#include <hpx/modules/iterator_support.hpp>
#include <hpx/modules/testing.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
#define LOOP_SMALL 10000
#define SKIP_SMALL 1000

#define LOOP_LARGE 100
#define SKIP_LARGE 10

#define LARGE_MESSAGE_SIZE 8192

#define MAX_MSG_SIZE (1 << 22)
#define MAX_ALIGNMENT 65536
#define SEND_BUFSIZE (MAX_MSG_SIZE + MAX_ALIGNMENT)

char send_buffer[SEND_BUFSIZE];

///////////////////////////////////////////////////////////////////////////////
char* align_buffer(char* ptr, unsigned long align_size)
{
    return (char*) (((std::size_t) ptr + (align_size - 1)) / align_size *
        align_size);
}

#if defined(HPX_WINDOWS)
unsigned long getpagesize()
{
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return si.dwPageSize;
}
#endif

///////////////////////////////////////////////////////////////////////////////
hpx::serialization::serialize_buffer<char> isend(
    hpx::serialization::serialize_buffer<char> const& receive_buffer)
{
    return receive_buffer;
}
HPX_PLAIN_ACTION(isend)

///////////////////////////////////////////////////////////////////////////////
double ireceive(hpx::id_type dest, std::size_t size, std::size_t window_size)
{
    int loop = LOOP_SMALL;
    int skip = SKIP_SMALL;

    if (size > LARGE_MESSAGE_SIZE)
    {
        loop = LOOP_LARGE;
        skip = SKIP_LARGE;
    }

    // align used buffers on page boundaries
    unsigned long align_size = getpagesize();
    HPX_TEST_LTE(align_size, static_cast<unsigned long>(MAX_ALIGNMENT));

    char* aligned_send_buffer = align_buffer(send_buffer, align_size);
    std::memset(aligned_send_buffer, 'a', size);

    hpx::chrono::high_resolution_timer t;

    isend_action send;
    for (int i = 0; i != loop + skip; ++i)
    {
        // do not measure warm up phase
        if (i == skip)
            t.restart();

        typedef hpx::serialization::serialize_buffer<char> buffer_type;

        using hpx::execution::par;
        using hpx::ranges::for_each;

        auto range = hpx::util::counting_shape(window_size);
        for_each(par, range, [&](std::uint64_t) {
            send(dest,
                buffer_type(aligned_send_buffer, size, buffer_type::reference));
        });
    }

    double elapsed = t.elapsed();
    return (elapsed * 1e6) /
        static_cast<double>(static_cast<std::size_t>(2) * loop * window_size);
}
HPX_PLAIN_ACTION(ireceive)

///////////////////////////////////////////////////////////////////////////////
void print_header()
{
    std::cout << "# OSU HPX Multi Latency Test\n"
              << "# Size    Latency (microsec)" << std::endl;
}

///////////////////////////////////////////////////////////////////////////////
void run_benchmark(hpx::program_options::variables_map& vm)
{
    std::vector<hpx::id_type> localities = hpx::find_all_localities();
    std::size_t window_size = vm["window-size"].as<std::size_t>();
    std::size_t pairs = localities.size() / 2;

    for (std::size_t size = 1; size <= MAX_MSG_SIZE; size *= 2)
    {
        std::vector<hpx::future<double>> benchmarks;

        for (std::uint32_t locality_id = 0; locality_id != localities.size();
            ++locality_id)
        {
            ireceive_action receive;

            std::size_t partner = 0;
            if (locality_id < pairs)
                partner = locality_id + pairs;
            else
                partner = locality_id - pairs;

            benchmarks.push_back(hpx::async(receive, localities[locality_id],
                localities[partner], size, window_size));
        }

        double total_latency = std::accumulate(benchmarks.begin(),
            benchmarks.end(), 0.0,
            [](double sum, hpx::future<double>& f) { return sum + f.get(); });

        std::cout << std::left << std::setw(10) << size
                  << total_latency / (2. * static_cast<double>(pairs))
                  << std::endl;
    }
}
#endif
