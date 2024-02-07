//  Copyright (c) 2023 Jiakun Yan
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx_init.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/runtime.hpp>
#include <hpx/iostream.hpp>
#include <hpx/modules/timing.hpp>
#include <hpx/serialization.hpp>

#include <atomic>
#include <cstddef>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
const std::size_t nbytes_default = 8;
const std::size_t nsteps_default = 1;
const std::size_t window_default = 10000;
const std::size_t inject_rate_default = 0;
const std::size_t batch_size_default = 10;
const std::size_t nwarmups_default = 1;
const std::size_t niters_default = 1;

size_t window;
size_t inject_rate;
size_t batch_size;

///////////////////////////////////////////////////////////////////////////////

void set_window(std::size_t window);
HPX_PLAIN_ACTION(set_window, set_window_action)

void on_inject(hpx::id_type to, size_t nbytes, std::size_t nsteps);
HPX_PLAIN_ACTION(on_inject, on_inject_action)

void on_recv(hpx::id_type to, std::vector<char> const& in, std::size_t counter);
HPX_PLAIN_ACTION(on_recv, on_recv_action)

void on_done();
HPX_PLAIN_ACTION(on_done, on_done_action)

void set_window(std::size_t window_)
{
    window = window_;
}

void on_inject(hpx::id_type to, std::size_t nbytes, std::size_t nsteps)
{
    hpx::chrono::high_resolution_timer timer;
    for (size_t i = 0; i < batch_size; ++i)
    {
        while (inject_rate > 0 &&
            static_cast<double>(i) / timer.elapsed() >
                static_cast<double>(inject_rate))
        {
            hpx::this_thread::yield();
        }
        std::vector<char> data(nbytes, 'a');
        hpx::post<on_recv_action>(to, hpx::find_here(), data, nsteps);
    }
}

std::atomic<size_t> done_counter(0);

void on_recv(hpx::id_type to, std::vector<char> const& in, std::size_t counter)
{
    // received vector in
    if (--counter == 0)
    {
        size_t result = done_counter.fetch_add(1, std::memory_order_relaxed);
        if (result + 1 == window)
        {
            hpx::post<on_done_action>(hpx::find_root_locality());
            done_counter = 0;
        }
        return;
    }

    // send it to remote locality (to)
    std::vector<char> data(in);
    hpx::post<on_recv_action>(to, hpx::find_here(), std::move(data), counter);
}

hpx::counting_semaphore_var<> semaphore;

void on_done()
{
    semaphore.signal();
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(hpx::program_options::variables_map& b_arg)
{
    std::size_t const nbytes = b_arg["nbytes"].as<std::size_t>();
    std::size_t const nsteps = b_arg["nsteps"].as<std::size_t>();
    bool verbose = b_arg["verbose"].as<bool>();
    window = b_arg["window"].as<std::size_t>();
    inject_rate = b_arg["inject-rate"].as<std::size_t>();
    batch_size = b_arg["batch-size"].as<std::size_t>();
    std::size_t const nwarmups = b_arg["nwarmups"].as<std::size_t>();
    std::size_t const niters = b_arg["niters"].as<std::size_t>();

    if (nsteps == 0)
    {
        std::cout << "nsteps is 0!" << std::endl;
        return 0;
    }

    if (window == 0)
    {
        std::cout << "window is 0!" << std::endl;
        return 0;
    }

    std::vector<hpx::id_type> localities = hpx::find_remote_localities();

    hpx::id_type to;
    if (localities.size() == 0)
    {
        to = hpx::find_here();
    }
    else
    {
        to = localities[0];    // send to first remote locality
    }

    set_window_action act;
    act(to, window);

    double inject_time = 0;
    double time = 0;
    for (size_t j = 0; j < nwarmups + niters; ++j)
    {
        hpx::chrono::high_resolution_timer timer_total;

        for (size_t i = 0; i < window; i += batch_size)
        {
            while (inject_rate > 0 &&
                static_cast<double>(i) / timer_total.elapsed() >
                    static_cast<double>(inject_rate))
            {
                continue;
            }
            hpx::post<on_inject_action>(hpx::find_here(), to, nbytes, nsteps);
        }
        if (j >= nwarmups)
            inject_time += timer_total.elapsed();

        semaphore.wait();
        if (j >= nwarmups)
            time += timer_total.elapsed();
    }

    double achieved_inject_rate =
        static_cast<double>(window * niters) / inject_time / 1e3;
    double latency = time * 1e6 / static_cast<double>(nsteps * niters);
    double msg_rate =
        static_cast<double>(nsteps * window * niters) / time / 1e3;
    double bandwidth =
        static_cast<double>(nbytes * nsteps * window * niters) / time / 1e6;
    if (verbose)
    {
        std::cout << "[hpx_pingpong]" << std::endl
                  << "total_time(secs)=" << time << std::endl
                  << "nwarmups=" << nwarmups << std::endl
                  << "niters=" << niters << std::endl
                  << "nbytes=" << nbytes << std::endl
                  << "window=" << window << std::endl
                  << "latency(us)=" << latency << std::endl
                  << "inject_rate(K/s)=" << achieved_inject_rate << std::endl
                  << "msg_rate(K/s)=" << msg_rate << std::endl
                  << "bandwidth(MB/s)=" << bandwidth << std::endl
                  << "localities=" << localities.size() << std::endl
                  << "nsteps=" << nsteps << std::endl;
    }
    else
    {
        std::cout << "[hpx_pingpong]"
                  << ":total_time(secs)=" << time << ":nbytes=" << nbytes
                  << ":nwarmups=" << nwarmups << ":niters=" << niters
                  << ":window=" << window << ":latency(us)=" << latency
                  << ":inject_rate(K/s)=" << achieved_inject_rate
                  << ":msg_rate(M/s)=" << msg_rate
                  << ":bandwidth(MB/s)=" << bandwidth
                  << ":localities=" << localities.size() << ":nsteps=" << nsteps
                  << std::endl;
    }

    hpx::finalize();
    return 0;
}

int main(int argc, char* argv[])
{
    namespace po = hpx::program_options;
    po::options_description description("HPX pingpong example");

    description.add_options()("nbytes",
        po::value<std::size_t>()->default_value(nbytes_default),
        "number of elements (doubles) to send/receive (integer)")("nsteps",
        po::value<std::size_t>()->default_value(nsteps_default),
        "number of ping-pong iterations")("window",
        po::value<std::size_t>()->default_value(window_default),
        "window size of ping-pong")("inject-rate",
        po::value<std::size_t>()->default_value(inject_rate_default),
        "the rate of injecting the first message of ping-pong")("batch-size",
        po::value<std::size_t>()->default_value(batch_size_default),
        "the number of messages to inject per inject thread")("nwarmups",
        po::value<std::size_t>()->default_value(nwarmups_default),
        "the iteration count of warmup runs")("niters",
        po::value<std::size_t>()->default_value(niters_default),
        "the iteration count of measurement iterations.")("verbose",
        po::value<bool>()->default_value(true),
        "verbosity of output,if false output is for awk");

    hpx::init_params init_args;
    init_args.desc_cmdline = description;

    return hpx::init(argc, argv, init_args);
}

#endif
