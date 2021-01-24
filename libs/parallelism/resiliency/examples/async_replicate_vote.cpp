//  Copyright (c) 2019 National Technology & Engineering Solutions of Sandia,
//                     LLC (NTESS).
//  Copyright (c) 2019 Nikunj Gupta
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/future.hpp>
#include <hpx/modules/resiliency.hpp>
#include <hpx/modules/timing.hpp>

#include <cstddef>
#include <iostream>
#include <random>
#include <unordered_map>
#include <vector>

std::random_device rd;
std::mt19937 mt(rd());
std::uniform_real_distribution<double> dist(1.0, 10.0);

int vote(std::vector<int> vect)
{
    return vect.at(0);
}

int universal_ans()
{
    if (dist(mt) > 5)
        return 42;
    return 84;
}

bool validate(int ans)
{
    return ans == 42;
}

int hpx_main(hpx::program_options::variables_map& vm)
{
    std::size_t n = vm["n-value"].as<std::size_t>();

    {
        // Initialize a high resolution timer
        hpx::chrono::high_resolution_timer t;

        hpx::future<int> f =
            hpx::resiliency::experimental::async_replicate_vote(
                n, &vote, &universal_ans);

        std::cout << "Universal ans (maybe true): " << f.get() << std::endl;

        double elapsed = t.elapsed();
        hpx::util::format_to(std::cout, "Time elapsed == {1}\n", elapsed);
    }

    {
        // Initialize a high resolution timer
        hpx::chrono::high_resolution_timer t;

        hpx::future<int> f =
            hpx::resiliency::experimental::async_replicate_vote_validate(
                n, &vote, &validate, &universal_ans);

        std::cout << "Universal ans (true ans): " << f.get() << std::endl;

        double elapsed = t.elapsed();
        hpx::util::format_to(std::cout, "Time elapsed == {1}\n", elapsed);
    }

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    using hpx::program_options::options_description;
    using hpx::program_options::value;

    // Configure application specific options
    options_description desc_commandline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()("n-value",
        value<std::size_t>()->default_value(10),
        "Number of asynchronous function launches (curated for successful "
        "replicate example)");

    // Initialize and run HPX
    hpx::init_params init_args;
    init_args.desc_cmdline = desc_commandline;

    return hpx::init(argc, argv, init_args);
}
