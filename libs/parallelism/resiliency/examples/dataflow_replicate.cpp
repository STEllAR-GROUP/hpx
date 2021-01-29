//  Copyright (c) 2019 National Technology & Engineering Solutions of Sandia,
//                     LLC (NTESS).
//  Copyright (c) 2018 Hartmut Kaiser
//  Copyright (c) 2019 Adrian Serio
//  Copyright (c) 2019 Nikunj Gupta
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/modules/resiliency.hpp>

#include <atomic>
#include <iostream>
#include <random>
#include <stdexcept>

std::mt19937 gen(1);

struct bad_calc_exception : std::exception
{
};
struct inv_arg_exception : std::exception
{
};

int moody_calc(int a, int b)
{
    if (gen() % 2 == 0)
        return a + b;
    else
        return a;
}

int moody_add(int a, int b)
{
    if (a == 0 || b == 0)
        throw inv_arg_exception();
    int c = moody_calc(a, b);
    if (c == a)
        throw bad_calc_exception();
    else
        return c;
}

int bad_add(int, int)
{
    throw bad_calc_exception();
}

int hpx_main(hpx::program_options::variables_map& vm)
{
    std::size_t sr = vm["sr-value"].as<std::size_t>();
    std::size_t usr = vm["usr-value"].as<std::size_t>();
    std::size_t a = vm["a-value"].as<std::size_t>();

    {
        // Successful replicate
        hpx::future<int> f = hpx::resiliency::experimental::dataflow_replicate(
            sr, moody_add, 5, 5);
        try
        {
            std::cout << f.get() << std::endl;
        }
        catch (inv_arg_exception const&)
        {
            std::cout << "Invalid Argument!" << std::endl;
        }
        catch (bad_calc_exception const&)
        {
            std::cout << "Bad Calculation!" << std::endl;
        }

        // Unsuccessful replicate
        f = hpx::resiliency::experimental::dataflow_replay(
            usr, moody_add, 0, 5);
        try
        {
            std::cout << f.get() << std::endl;
        }
        catch (inv_arg_exception const&)
        {
            std::cout << "Invalid Argument!" << std::endl;
        }
        catch (bad_calc_exception const&)
        {
            std::cout << "Bad Calculation!" << std::endl;
        }

        // Aborted replicate
        f = hpx::resiliency::experimental::dataflow_replay(a, bad_add, 10, 5);
        try
        {
            std::cout << f.get() << std::endl;
        }
        catch (inv_arg_exception const&)
        {
            std::cout << "Invalid Argument!" << std::endl;
        }
        catch (bad_calc_exception const&)
        {
            std::cout << "Bad Calculation!" << std::endl;
        }
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

    desc_commandline.add_options()("sr-value",
        value<std::size_t>()->default_value(10),
        "Number of repeat launches for successful dataflow replicate");

    desc_commandline.add_options()("usr-value",
        value<std::size_t>()->default_value(6),
        "Number of repeat launches for unsuccessful dataflow replicate");

    desc_commandline.add_options()("a-value",
        value<std::size_t>()->default_value(3),
        "Number of repeat launches for aborted dataflow replicate");

    // Initialize and run HPX
    hpx::init_params init_args;
    init_args.desc_cmdline = desc_commandline;

    return hpx::init(argc, argv, init_args);
}
