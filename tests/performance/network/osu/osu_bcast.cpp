//  Copyright (c) 2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Broadcast network test

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/serialization.hpp>
#include <hpx/lcos/local/and_gate.hpp>
#include <hpx/util/any.hpp>

#include <boost/assert.hpp>
#include <boost/shared_ptr.hpp>

#include <benchmarks/network/osu_coll.hpp>
#include <benchmarks/network/broadcast.hpp>

HPX_PLAIN_ACTION(hpx::lcos::detail::broadcast_impl_action, broadcast_impl_action);


struct broadcast_component
  : hpx::components::simple_component_base<broadcast_component>
{
    broadcast_component()
    {}

    void init(std::vector<hpx::id_type> const & id, std::size_t max_msg_size, std::size_t fan_out)
    {
        bcast.this_id = this->get_id();
        bcast.fan_out = fan_out;
        ids = id;
        send_buffer = std::vector<char>(max_msg_size);
    }

    HPX_DEFINE_COMPONENT_ACTION(broadcast_component, init);

    typedef hpx::serialization::serialize_buffer<char> buffer_type;

    double run(std::size_t size, std::size_t iterations, std::size_t skip)
    {
        double elapsed = 0.0;
        for(std::size_t i = 0; i < iterations + skip; ++i)
        {
            hpx::util::high_resolution_timer t;

            recv_buffer = bcast(ids, 0, buffer_type(&send_buffer[0], size, buffer_type::reference)).get();

            double t_elapsed = t.elapsed();
            if(i >= skip)
            {
                elapsed += t_elapsed;
            }
        }

        double latency = (elapsed * 1e6) / iterations;

        return latency;
    }

    HPX_DEFINE_COMPONENT_ACTION(broadcast_component, run);

    HPX_DEFINE_COMPONENT_BROADCAST(bcast, buffer_type);
    std::vector<hpx::id_type> ids;
    std::vector<char> send_buffer;
    buffer_type recv_buffer;
};

HPX_REGISTER_COMPONENT(
    hpx::components::simple_component<broadcast_component>
  , osu_broadcast_component);


void run_benchmark(params const & p)
{
    std::size_t skip = SKIP;
    std::size_t iterations = p.iterations;

    std::vector<hpx::id_type> ids = create_components<broadcast_component>(p);

    if(ids.size() < 2)
    {
        hpx::cout << "This benchmark must be run with at least 2 threads" << hpx::endl << hpx::flush;
        return;
    }

    {
        std::vector<hpx::future<void> > init_futures;
        init_futures.reserve(ids.size());
        for (hpx::id_type const& id : ids)
        {
            init_futures.push_back(
                hpx::async<broadcast_component::init_action>(id, ids, p.max_msg_size, p.fan_out)
            );
        }
        hpx::wait_all(init_futures);
    }

    for(std::size_t size = 1; size <= p.max_msg_size; size *=2)
    {
        if(size > LARGE_MESSAGE_SIZE)
        {
            skip = SKIP_LARGE;
            iterations = ITERATIONS_LARGE;
        }

        std::vector<hpx::future<double> > run_futures;
        run_futures.reserve(ids.size());
        for (hpx::id_type const& id : ids)
        {
            run_futures.push_back(
                hpx::async<broadcast_component::run_action>(id, size, iterations, skip)
            );
        }


        std::vector<double> times; times.reserve(ids.size());
        hpx::wait_all(run_futures);
        for (hpx::future<double>& f : run_futures)
        {
            times.push_back(f.get());
        }

        double avg_latency = std::accumulate(times.begin(), times.end(), 0.0) / ids.size();

        print_data(avg_latency, size, iterations);
    }
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map & vm)
{
    params p(process_args(vm));
    print_header("OSU HPX Broadcast Latency Test");
    run_benchmark(p);
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    boost::program_options::options_description
        desc(params_desc());

    return hpx::init(desc, argc, argv);
}
