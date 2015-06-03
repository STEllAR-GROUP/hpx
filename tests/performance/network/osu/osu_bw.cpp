//  Copyright (c) 2013 Hartmut Kaiser
//  Copyright (c) 2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Unidirectional network bandwidth test

#include <hpx/hpx_main.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/runtime/serialization/serialize_buffer.hpp>
#include <hpx/parallel/algorithm.hpp>

#include <boost/assert.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/range/irange.hpp>

///////////////////////////////////////////////////////////////////////////////
char* align_buffer (char* ptr, unsigned long align_size)
{
    return (char*)(((std::size_t)ptr + (align_size - 1)) / align_size * align_size);
}

#if defined(BOOST_MSVC)
unsigned long getpagesize()
{
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return si.dwPageSize;
}
#endif

///////////////////////////////////////////////////////////////////////////////

#define LOOP_SMALL_MULTIPLIER 5
#define SKIP  2

#define LARGE_MESSAGE_SIZE  8192

#define MAX_MSG_SIZE (1<<22)
#define MAX_ALIGNMENT 65536


///////////////////////////////////////////////////////////////////////////////
void isend(hpx::serialization::serialize_buffer<char> const& receive_buffer) {}
HPX_PLAIN_ACTION(isend);

///////////////////////////////////////////////////////////////////////////////
double ireceive(hpx::naming::id_type dest, std::size_t loop,
                std::size_t size, std::size_t window_size)
{
    std::size_t skip = SKIP;

    if (size <= LARGE_MESSAGE_SIZE) {
        loop *= LOOP_SMALL_MULTIPLIER;
        skip *= LOOP_SMALL_MULTIPLIER;
    }

    typedef hpx::serialization::serialize_buffer<char> buffer_type;

    // align used buffers on page boundaries
    unsigned long align_size = getpagesize();
    (void)align_size;
    BOOST_ASSERT(align_size <= MAX_ALIGNMENT);

    char *send_buffer = new char[size];
    std::memset(send_buffer, 'a', size);

    hpx::util::high_resolution_timer t;

    isend_action send;
    for (std::size_t i = 0; i != loop + skip; ++i) {
        // do not measure warm up phase
        if (i == skip)
            t.restart();

        using hpx::parallel::for_each;
        using hpx::parallel::par;

        std::size_t const start = 0;

        // Fill the original matrix, set transpose to known garbage value.
        auto range = boost::irange(start, window_size);
        for_each(par, boost::begin(range), boost::end(range),
            [&](boost::uint64_t j)
            {
                send(dest,
                    buffer_type(send_buffer, size, buffer_type::reference));
            }
        );
    }

    double elapsed = t.elapsed();

    delete[] send_buffer;

    return (size / 1e6 * loop * window_size) / elapsed;
}

///////////////////////////////////////////////////////////////////////////////
void print_header ()
{
    hpx::cout << "# OSU HPX Bandwidth Test\n"
              << "# Size    Bandwidth (MB/s)\n"
              << hpx::flush;
}

void run_benchmark(boost::program_options::variables_map & vm)
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
        double bw = ireceive(there, loop, size, vm["window-size"].as<std::size_t>());
        hpx::cout << std::left << std::setw(10) << size
                  << bw << hpx::endl << hpx::flush;
    }
}
