//  Copyright (c) 2013-2015 Hartmut Kaiser
//  Copyright (c) 2013 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Bidirectional network bandwidth test

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/algorithm.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/serialization.hpp>
#include <hpx/modules/program_options.hpp>

#include <boost/range/irange.hpp>

#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
#define MESSAGE_ALIGNMENT 64
#define MAX_ALIGNMENT 65536
#define MAX_MSG_SIZE 8    // (1<<22)
#define SEND_BUFSIZE (MAX_MSG_SIZE + MAX_ALIGNMENT)

#define LOOP_LARGE 100
#define SKIP_LARGE 10

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
hpx::serialization::serialize_buffer<char> message(
    hpx::serialization::serialize_buffer<char> const& receive_buffer)
{
    return receive_buffer;
}
HPX_PLAIN_DIRECT_ACTION(message);

HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION(
    hpx::serialization::serialize_buffer<char>, serialization_buffer_char);
HPX_REGISTER_BASE_LCO_WITH_VALUE(
    hpx::serialization::serialize_buffer<char>, serialization_buffer_char);

double message_double(double d)
{
    return d;
}
HPX_PLAIN_DIRECT_ACTION(message_double);

///////////////////////////////////////////////////////////////////////////////
double receive_double(
    hpx::naming::id_type dest, std::size_t loop, std::size_t window_size)
{
    std::size_t skip = SKIP_LARGE;

    hpx::chrono::high_resolution_timer t;

    message_double_action msg;
    for (std::size_t i = 0; i != loop + skip; ++i)
    {
        // do not measure warm up phase
        if (i == skip)
            t.restart();

        using hpx::execution::par;
        using hpx::ranges::for_each;

        std::size_t const start = 0;

        auto range = boost::irange(start, window_size);
        for_each(par, range, [&](std::uint64_t) {
            double d = 0.0;
            msg(dest, d);
        });
    }

    double elapsed = t.elapsed();
    return (elapsed * 1e6) / static_cast<double>(2 * loop * window_size);
}
double receive(hpx::naming::id_type dest, char* send_buffer, std::size_t size,
    std::size_t loop, std::size_t window_size)
{
    std::size_t skip = SKIP_LARGE;

    typedef hpx::serialization::serialize_buffer<char> buffer_type;
    buffer_type recv_buffer;

    hpx::chrono::high_resolution_timer t;

    message_action msg;
    for (std::size_t i = 0; i != loop + skip; ++i)
    {
        // do not measure warm up phase
        if (i == skip)
            t.restart();

        using hpx::for_each;
        using hpx::execution::par;

        std::size_t const start = 0;

        auto range = boost::irange(start, window_size);
        for_each(par, std::begin(range), std::end(range), [&](std::uint64_t) {
            msg(dest, buffer_type(send_buffer, size, buffer_type::reference));
        });
    }

    double elapsed = t.elapsed();
    return (elapsed * 1e6) / static_cast<double>(2 * loop * window_size);
}

///////////////////////////////////////////////////////////////////////////////
void print_header()
{
    std::cout << "# OSU HPX Latency Test\n"
              << "# Size    Latency (microsec)" << std::endl;
}

///////////////////////////////////////////////////////////////////////////////
void run_benchmark(hpx::program_options::variables_map& vm)
{
    // use the first remote locality to bounce messages, if possible
    hpx::id_type here = hpx::find_here();

    hpx::id_type there = here;
    std::vector<hpx::id_type> localities = hpx::find_remote_localities();
    if (!localities.empty())
        there = localities[0];

    std::size_t window_size = vm["window-size"].as<std::size_t>();
    std::size_t loop = vm["loop"].as<std::size_t>();
    std::size_t min_size = vm["min-size"].as<std::size_t>();
    std::size_t max_size = vm["max-size"].as<std::size_t>();

    if (max_size < min_size)
        std::swap(max_size, min_size);

    // align used buffers on page boundaries
    unsigned long align_size = getpagesize();
    std::unique_ptr<char[]> send_buffer_orig(new char[max_size + align_size]);
    char* send_buffer = align_buffer(send_buffer_orig.get(), align_size);

    // perform actual measurements
    hpx::chrono::high_resolution_timer timer;

    // test for single double
    double latency = receive_double(there, loop, window_size);
    std::cout << std::left << std::setw(10) << "single double " << latency
              << std::endl;

    for (std::size_t size = min_size; size <= max_size; size *= 2)
    {
        double latency = receive(there, send_buffer, size, loop, window_size);
        std::cout << std::left << std::setw(10) << size << latency << std::endl;
    }

    std::cout << "Total time: " << timer.elapsed_nanoseconds() << std::endl;
}
#endif
