//  Copyright (c) 2013-2015 Hartmut Kaiser
//  Copyright (c) 2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Multi latency network test

#include <hpx/hpx.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/include/serialization.hpp>
#include <hpx/include/parallel_for_each.hpp>
#include <hpx/include/util.hpp>

#include <boost/range/irange.hpp>

#include <algorithm>
#include <cstring>
#include <numeric>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
#define LOOP_SMALL  10000
#define SKIP_SMALL  1000

#define LOOP_LARGE  100
#define SKIP_LARGE  10

#define LARGE_MESSAGE_SIZE  8192

#define MAX_MSG_SIZE (1<<22)
#define MAX_ALIGNMENT 65536
#define SEND_BUFSIZE (MAX_MSG_SIZE + MAX_ALIGNMENT)

char send_buffer[SEND_BUFSIZE];

///////////////////////////////////////////////////////////////////////////////
char* align_buffer (char* ptr, unsigned long align_size)
{
    return (char*)(((std::size_t)ptr + (align_size - 1)) / align_size * align_size);
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
hpx::serialization::serialize_buffer<char>
isend(hpx::serialization::serialize_buffer<char> const& receive_buffer)
{
    return receive_buffer;
}
HPX_PLAIN_ACTION(isend);

///////////////////////////////////////////////////////////////////////////////
double ireceive(hpx::naming::id_type dest, std::size_t size, std::size_t window_size)
{
    int loop = LOOP_SMALL;
    int skip = SKIP_SMALL;

    if (size > LARGE_MESSAGE_SIZE) {
        loop = LOOP_LARGE;
        skip = SKIP_LARGE;
    }

    // align used buffers on page boundaries
    unsigned long align_size = getpagesize();
    HPX_ASSERT(align_size <= MAX_ALIGNMENT);

    char* aligned_send_buffer = align_buffer(send_buffer, align_size);
    std::memset(aligned_send_buffer, 'a', size);

    hpx::util::high_resolution_timer t;

    isend_action send;
    for (int i = 0; i != loop + skip; ++i) {
        // do not measure warm up phase
        if (i == skip)
            t.restart();

        typedef hpx::serialization::serialize_buffer<char> buffer_type;

        using hpx::parallel::for_each;
        using hpx::parallel::par;

        std::size_t const start = 0;

        auto range = boost::irange(start, window_size);
        for_each(par, boost::begin(range), boost::end(range),
            [&](boost::uint64_t j)
            {
                send(dest, buffer_type(aligned_send_buffer, size,
                        buffer_type::reference));
            });
    }

    double elapsed = t.elapsed();
    return (elapsed * 1e6) / (2 * loop * window_size);
}
HPX_PLAIN_ACTION(ireceive);

///////////////////////////////////////////////////////////////////////////////
void print_header()
{
    hpx::cout << "# OSU HPX Multi Latency Test\n"
              << "# Size    Latency (microsec)"
              << std::endl;
}

///////////////////////////////////////////////////////////////////////////////
void run_benchmark(boost::program_options::variables_map & vm)
{
    std::vector<hpx::id_type> localities = hpx::find_all_localities();
    std::size_t window_size = vm["window-size"].as<std::size_t>();
    std::size_t pairs = localities.size() / 2;

    for (std::size_t size = 1; size <= MAX_MSG_SIZE; size *= 2)
    {
        std::vector<hpx::future<double> > benchmarks;

        for (boost::uint32_t locality_id = 0; locality_id !=
            localities.size(); ++locality_id)
        {
            ireceive_action receive;

            std::size_t partner = 0;
            if(locality_id < pairs)
                partner = locality_id + pairs;
            else
                partner = locality_id - pairs;

            benchmarks.push_back(hpx::async(receive,
                localities[locality_id], localities[partner], size,
                window_size));
        }

        double total_latency = std::accumulate(
            benchmarks.begin(), benchmarks.end(), 0.0,
            [](double sum, hpx::future<double>& f)
            {
                return sum + f.get();
            });

        hpx::cout << std::left << std::setw(10) << size
                  << total_latency / (2. * pairs)
                  << hpx::endl << hpx::flush;
    }
}
