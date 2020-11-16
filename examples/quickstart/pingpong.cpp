//  Copyright (c) 2012 Mehmet Balman
//  Copyright (c) 2012 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx_init.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/components.hpp>
#include <hpx/iostream.hpp>
#include <hpx/serialization.hpp>
#include <hpx/modules/timing.hpp>

#include <cstddef>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
const std::size_t vsize_default = 1024*1024;
const std::size_t numiter_default = 5;

///////////////////////////////////////////////////////////////////////////////
void on_recv(hpx::naming::id_type to, std::vector<double> const & in,
    std::size_t counter);
HPX_PLAIN_ACTION(on_recv, on_recv_action);

void on_recv(hpx::naming::id_type to, std::vector<double> const & in,
    std::size_t counter)
{
    // received vector in
    if (--counter == 0) return;


    // send it to remote locality (to), and wait until it is received
    std::vector<double> data(in);

    on_recv_action act;
    act(to, hpx::find_here(), std::move(data), counter);
}

///////////////////////////////////////////////////////////////////////////////
void on_recv_ind(hpx::naming::id_type to,
    std::shared_ptr<std::vector<double> > const& in, std::size_t counter);
HPX_PLAIN_ACTION(on_recv_ind, on_recv_ind_action);

void on_recv_ind(hpx::naming::id_type to,
    std::shared_ptr<std::vector<double> > const& in, std::size_t counter)
{
    // received vector in
    if (--counter == 0) return;

    // send it to remote locality (to), and wait until it is received
    std::shared_ptr<std::vector<double> > data(
        std::make_shared<std::vector<double> >(*in));

    on_recv_ind_action act;
    act(to, hpx::find_here(), data, counter);
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(hpx::program_options::variables_map &b_arg)
{
    std::size_t const vsize = b_arg["vsize"].as<std::size_t>();
    std::size_t const numiter = b_arg["numiter"].as<std::size_t>() * 2;
    bool verbose = b_arg["verbose"].as<bool>();

    std::vector<hpx::naming::id_type> localities = hpx::find_remote_localities();

    hpx::naming::id_type to;
    if(localities.size() == 0)
    {
        to = hpx::find_here();
    }
    else
    {
        to = localities[0]; // send to first remote locality
    }

    // test sending messages back and forth using a larger vector as one of
    // the arguments
    {
        std::vector<double> data(vsize, double(3.11));

        hpx::chrono::high_resolution_timer timer1;

        if (numiter != 0) {
            on_recv_action act;
            act(to, hpx::find_here(), data, numiter);
        }

        double time = timer1.elapsed();
        double bandwidth =
            ((static_cast<double>(vsize * sizeof(double) * numiter) / time) /
                1024) /
            1024;
        if (verbose) {
            std::cout << "[hpx_pingpong]" << std::endl
                      << "total_time(secs)=" << time << std::endl
                      << "vsize=" << vsize << " = " << vsize * sizeof(double)
                      << " Bytes" << std::endl
                      << "bandwidth(MiB/s)=" << bandwidth << std::endl
                      << "localities=" << localities.size() << std::endl
                      << "numiter=" << numiter << std::endl;
        }
        else {
            std::cout << "[hpx_pingpong]"
                      << ":total_time(secs)=" << time << ":vsize=" << vsize
                      << ":bandwidth(MiB/s)=" << bandwidth
                      << ":localities=" << localities.size()
                      << ":numiter=" << numiter << std::endl;
        }
    }

    // do the same but with a wrapped vector
    /*
    {
        std::shared_ptr<std::vector<double> > data(
            std::make_shared<std::vector<double> >(vsize, double(3.11)));

        hpx::chrono::high_resolution_timer timer1;

        if (numiter != 0) {
            on_recv_ind_action act;
            act(to, hpx::find_here(), data, numiter);
        }

        double time = timer1.elapsed();

        std::cout << "[hpx_pingpong]"
                  << ":total_time(secs)=" << time
                  << ":vsize=" << vsize
                  << ":bandwidth(GB/s)=" << (vsize * sizeof(double) * numiter)
                    / (time * 1024 * 1024)
                  << ":localities=" << localities.size()
                  << ":numiter=" << numiter << std::endl;
    }
    */

    hpx::finalize();
    return 0;
}

int main(int argc, char* argv[])
{
    namespace po = hpx::program_options;
    po::options_description description("HPX pingpong example");

    description.add_options()
        ( "vsize", po::value<std::size_t>()->default_value(vsize_default),
          "number of elements (doubles) to send/receive  (integer)")
        ( "numiter", po::value<std::size_t>()->default_value(numiter_default),
          "number of ping-pong iterations")
        ( "verbose", po::value<bool>()->default_value(true),
         "verbosity of output,if false output is for awk")
        ;

    hpx::init_params init_args;
    init_args.desc_cmdline = description;

    return hpx::init(argc, argv, init_args);
}

#endif
