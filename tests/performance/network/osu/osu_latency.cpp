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
#include <hpx/lcos/local/detail/sliding_semaphore.hpp>

#include <boost/scoped_array.hpp>
#include <boost/range/irange.hpp>

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
#define MESSAGE_ALIGNMENT 64
#define MAX_ALIGNMENT 65536
#define MAX_MSG_SIZE 8 // (1<<22)
#define SEND_BUFSIZE (MAX_MSG_SIZE + MAX_ALIGNMENT)

#define LOOP_LARGE  100
#define SKIP_LARGE  10

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
message(hpx::serialization::serialize_buffer<char> const& receive_buffer)
{
    return receive_buffer;
}
HPX_PLAIN_ACTION(message);

HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION(
    hpx::serialization::serialize_buffer<char>, serialization_buffer_char);
HPX_REGISTER_BASE_LCO_WITH_VALUE(
    hpx::serialization::serialize_buffer<char>, serialization_buffer_char);

double
message_double(double d)
{
    return d;
}
HPX_PLAIN_ACTION(message_double);

///////////////////////////////////////////////////////////////////////////////
double receive_double(
    hpx::naming::id_type dest,
    std::size_t loop,
    std::size_t window_size)
{
    std::size_t skip = SKIP_LARGE;


    hpx::util::high_resolution_timer t;

    message_double_action msg;
    for (std::size_t i = 0; i != loop + skip; ++i) {
        // do not measure warm up phase
        if (i == skip)
            t.restart();

        using hpx::parallel::for_each;
        using hpx::parallel::par;

        std::size_t const start = 0;

        auto range = boost::irange(start, window_size);
        for_each(par, boost::begin(range), boost::end(range),
            [&](std::uint64_t j)
            {
                double d = 0.0;
                msg(dest, d);
            }
        );
    }

    double elapsed = t.elapsed();
    return (elapsed * 1e6) / (2 * loop * window_size);
}

double receive(
    hpx::naming::id_type dest,
    char * send_buffer,
    std::size_t size,
    std::size_t loop,
    std::size_t window_size)
{
    std::size_t skip = SKIP_LARGE;

    typedef hpx::serialization::serialize_buffer<char> buffer_type;
    buffer_type recv_buffer;

    hpx::util::high_resolution_timer t;

    message_action msg;
    hpx::lcos::local::sliding_semaphore sem(window_size);
    int parcel_count = 0;
    //
    int divisor = (window_size>1) ? (window_size/2) : 1;
    //
    for (std::size_t i = 0; i != (loop*window_size) + skip; ++i) {
        // do not measure warm up phase
        if (i == skip)
            t.restart();

        // launch a message to the remote node
        if (parcel_count % divisor == 0) {

            hpx::async(msg, dest,
                buffer_type(send_buffer, size, buffer_type::reference)).then(
                    hpx::launch::sync,
                    // when the message completes, increment our semaphore count
                    // so that N are always in flight
                    [&,parcel_count](auto &&f) -> void {
                        sem.signal(parcel_count);
                    }
                );
        }
        else {
            hpx::async(msg, dest,
                buffer_type(send_buffer, size, buffer_type::reference));
        }
        //
        sem.wait(parcel_count);
        //
        parcel_count++;
    }
    // when we reach this point, there will still be "window_size" parcels in flight
    // and we skipped some at the start, so take that into account in our
    // calculation of speed/latency

    double d = (static_cast<double>(parcel_count) - (window_size + skip));
    double elapsed = t.elapsed();
    return 0.5*(elapsed * 1e6) / d;
}

///////////////////////////////////////////////////////////////////////////////
void print_header()
{
    hpx::cout << "# OSU HPX Latency Test\n"
              << "# Size    Latency (microsec)"
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

    std::size_t window_size = vm["window-size"].as<std::size_t>();
    std::size_t loop = vm["loop"].as<std::size_t>();
    std::size_t min_size = vm["min-size"].as<std::size_t>();
    std::size_t max_size = vm["max-size"].as<std::size_t>();

    if(max_size < min_size) std::swap(max_size, min_size);

    // align used buffers on page boundaries
    unsigned long align_size = getpagesize();
    boost::scoped_array<char> send_buffer_orig(new char[max_size + align_size]);
    char* send_buffer = align_buffer(send_buffer_orig.get(), align_size);

    // perform actual measurements
    hpx::util::high_resolution_timer timer;

    // test for single double
    double latency = receive_double(there, loop, window_size);
    hpx::cout << std::left << std::setw(10) << "single double "
              << latency << hpx::endl << hpx::flush;

    for (std::size_t size = min_size; size <= max_size; size *= 2)
    {
        double latency = receive(there, send_buffer, size, loop, window_size);
        hpx::cout << std::left << std::setw(10) << size
                  << latency << hpx::endl << hpx::flush;
    }

    hpx::cout << "Total time: " << timer.elapsed_nanoseconds() << std::endl;
}
