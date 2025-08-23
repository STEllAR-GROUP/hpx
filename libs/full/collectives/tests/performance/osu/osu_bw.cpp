//  Copyright (c) 2013-2025 Hartmut Kaiser
//  Copyright (c) 2013 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Unidirectional network bandwidth test

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/algorithm.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/serialization.hpp>
#include <hpx/include/util.hpp>
#include <hpx/modules/iterator_support.hpp>
#include <hpx/modules/testing.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

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
#define LOOP_SMALL_MULTIPLIER 5
#define SKIP 2

#define LARGE_MESSAGE_SIZE 8192

#define MAX_MSG_SIZE (1 << 22)
#define MAX_ALIGNMENT 65536

///////////////////////////////////////////////////////////////////////////////
void isend(hpx::serialization::serialize_buffer<char> const&) {}
HPX_PLAIN_ACTION(isend)

///////////////////////////////////////////////////////////////////////////////
double ireceive(hpx::id_type dest, std::size_t loop, std::size_t size,
    std::size_t window_size)
{
    std::size_t skip = SKIP;

    if (size <= LARGE_MESSAGE_SIZE)
    {
        loop *= LOOP_SMALL_MULTIPLIER;
        skip *= LOOP_SMALL_MULTIPLIER;
    }

    typedef hpx::serialization::serialize_buffer<char> buffer_type;

    // align used buffers on page boundaries
    unsigned long align_size = getpagesize();
    (void) align_size;
    HPX_TEST_LTE(align_size, static_cast<unsigned long>(MAX_ALIGNMENT));

    std::unique_ptr<char[]> send_buffer(new char[size]);
    std::memset(send_buffer.get(), 'a', size);

    hpx::chrono::high_resolution_timer t;

    isend_action send;
    for (std::size_t i = 0; i != loop + skip; ++i)
    {
        // do not measure warm up phase
        if (i == skip)
            t.restart();

        using hpx::execution::par;
        using hpx::ranges::for_each;

        auto range = hpx::util::counting_shape(window_size);
        for_each(par, range, [&](std::uint64_t) {
            send(dest,
                buffer_type(send_buffer.get(), size, buffer_type::reference));
        });
    }

    double elapsed = t.elapsed();
    return (static_cast<double>(size) / 1e6 *
               static_cast<double>(loop * window_size)) /
        elapsed;
}

///////////////////////////////////////////////////////////////////////////////
void print_header()
{
    std::cout << "# OSU HPX Bandwidth Test\n"
              << "# Size    Bandwidth (MB/s)" << std::endl;
}

void run_benchmark(hpx::program_options::variables_map& vm)
{
    // use the first remote locality to bounce messages, if possible
    hpx::id_type here = hpx::find_here();

    hpx::id_type there = here;
    std::vector<hpx::id_type> localities = hpx::find_remote_localities();
    if (!localities.empty())
        there = localities[0];

    std::size_t max_size = vm["max-size"].as<std::size_t>();
    std::size_t min_size = vm["min-size"].as<std::size_t>();
    std::size_t loop = vm["loop"].as<std::size_t>();

    // perform actual measurements
    for (std::size_t size = min_size; size <= max_size; size *= 2)
    {
        double bw =
            ireceive(there, loop, size, vm["window-size"].as<std::size_t>());
        std::cout << std::left << std::setw(10) << size << bw << std::endl;
    }
}
#endif
