//  Copyright (c) 2012 Mehmet Balman
//  Copyright (c) 2012 Aydin Buluc
//  Copyright (c) 2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/iostreams.hpp>

#include <vector>

const std::size_t vsize_default = 1024*1024;
const std::size_t numiter_default = 5;

void onRecv(hpx::naming::id_type to, std::vector<double> const & in, std::size_t counter);
HPX_PLAIN_ACTION(onRecv, onRecv_action);

void onRecv(hpx::naming::id_type to, std::vector<double> const & in, std::size_t counter)
{
    // received vector in
    if (--counter == 0) return;

    // send it to remote locality (to), and wait till it is received
    std::vector<double> mydata(in);

    onRecv_action act;
    act(to, hpx::find_here(), mydata, counter);
}

int hpx_main(boost::program_options::variables_map &b_arg)
{
    std::size_t const vsize = b_arg["vsize"].as<std::size_t>();
    std::size_t const numiter = b_arg["numiter"].as<std::size_t>() * 2;

    std::vector<hpx::naming::id_type> localities = hpx::find_all_localities();
    hpx::naming::id_type to = localities.back(); // send to last element

    std::vector<double> mydata(vsize, double(3.11));

    hpx::util::high_resolution_timer timer1;

    if (numiter != 0) {
        onRecv_action act;
        act(to, hpx::find_here(), mydata, numiter);
    }

    double time = timer1.elapsed();

    std::cout << "[hpx_pingpong]:total_time(secs)=" << time << ":vsize="
              << vsize << ":localities=" << localities.size()
              << ":numiter=" << numiter << std::endl;

    hpx::finalize();
    return 0;
}

int main(int argc, char* argv[])
{
    namespace po = boost::program_options;
    po::options_description description("HPX pingpong example");

    description.add_options()
        ( "vsize", po::value<std::size_t>()->default_value(vsize_default),
          "number of elements (doubles) to send/receive  (integer)")
        ( "numiter", po::value<std::size_t>()->default_value(numiter_default),
          "number of ping-pong iterations")
        ;

    return hpx::init(description, argc, argv);
}

