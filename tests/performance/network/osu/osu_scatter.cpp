//  Copyright (c) 2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Scatter network test

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/runtime/serialization/serialize_buffer.hpp>

#include <boost/assert.hpp>

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include <benchmarks/network/osu_coll.hpp>

void scatter(std::vector<hpx::id_type> const & localities,
    hpx::serialization::serialize_buffer<char> buffer, std::size_t chunk_size);
HPX_PLAIN_ACTION(scatter);

void scatter(std::vector<hpx::id_type> const & localities,
    hpx::serialization::serialize_buffer<char> buffer, std::size_t chunk_size)
{
    std::vector<hpx::future<void> > scatter_futures;
    scatter_futures.reserve(localities.size() / chunk_size);

    typedef std::vector<hpx::id_type>::const_iterator iterator;
    iterator begin = localities.cbegin() + 1;

    if(localities.size() > 1)
    {
        for(std::size_t i = 0; i < chunk_size; ++i)
        {
            iterator end
                = (i == chunk_size-1)
                ? localities.cend()
                : begin + (localities.size() - 1)/chunk_size + 1;

            std::vector<hpx::id_type> locs(begin, end);
            if(locs.size() > 0)
            {
                hpx::id_type dst = locs[0];

                scatter_futures.push_back(
                    hpx::async<scatter_action>(dst, std::move(locs), buffer, chunk_size)
                );
            }

            begin = end;
        }
    }

    // Call some action for this locality here ...

    if(scatter_futures.size() > 0)
    {
        hpx::wait_all(scatter_futures);
    }
}

void run_benchmark(params const & p)
{
    std::size_t skip = SKIP;
    std::size_t iterations = p.iterations;

    std::vector<hpx::id_type> localities = hpx::find_all_localities();

    if(localities.size() < 2)
    {
        hpx::cout << "This benchmark must be run with at least 2 localities"
                  << hpx::endl << hpx::flush;
        return;
    }

    std::vector<char> send_buffer(p.max_msg_size);

    for(std::size_t size = 1; size <= p.max_msg_size; size *=2)
    {
        if(size > LARGE_MESSAGE_SIZE)
        {
            skip = SKIP_LARGE;
            iterations = ITERATIONS_LARGE;
        }


        double elapsed = 0.0;
        for(std::size_t i = 0; i < iterations + skip; ++i)
        {
            hpx::util::high_resolution_timer t;
            typedef hpx::serialization::serialize_buffer<char> buffer_type;
            hpx::id_type dst = localities[0];
            scatter_action()(dst, localities, buffer_type(&send_buffer[0],
                size, buffer_type::reference), p.chunk_size);
            double t_elapsed = t.elapsed();
            if(i >= skip)
                elapsed += t_elapsed;
        }

        print_data(elapsed, size, iterations);
    }
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map & vm)
{
    params p(process_args(vm));
    print_header("OSU HPX Scatter Latency Test");
    run_benchmark(p);
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    boost::program_options::options_description
        desc(params_desc());

    return hpx::init(desc, argc, argv);
}
