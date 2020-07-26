//  Copyright (c) 2019 National Technology & Engineering Solutions of Sandia,
//                     LLC (NTESS).
//  Copyright (c) 2018-2020 Hartmut Kaiser
//  Copyright (c) 2019 Adrian Serio
//  Copyright (c) 2019 Nikunj Gupta
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/modules/resiliency.hpp>
#include <hpx/modules/timing.hpp>

#include <atomic>
#include <iostream>
#include <stdexcept>

std::atomic<int> answer(35);

struct vogon_exception : std::exception
{
};

int universal_answer()
{
    return ++answer;
}

bool validate(int result)
{
    return result == 42;
}

int no_answer()
{
    throw hpx::resiliency::experimental::abort_replicate_exception();
}

int deep_thought()
{
    static int ans = 35;
    ++ans;
    if (ans == 42)
        return ans;
    else
        throw vogon_exception();
}

int hpx_main(hpx::program_options::variables_map& vm)
{
    std::size_t sr = vm["sr-value"].as<std::size_t>();
    std::size_t usr = vm["usr-value"].as<std::size_t>();
    std::size_t a = vm["a-value"].as<std::size_t>();

    {
        // Initialize a high resolution timer
        hpx::util::high_resolution_timer t;

        // successful replicate
        hpx::future<int> f =
            hpx::resiliency::experimental::async_replicate(sr, &deep_thought);

        std::cout << "universal answer: " << f.get() << "\n";

        // successful replicate_validate
        f = hpx::resiliency::experimental::async_replicate_validate(
            sr, &validate, &universal_answer);

        std::cout << "universal answer: " << f.get() << "\n";

        // unsuccessful replicate
        f = hpx::resiliency::experimental::async_replicate(usr, &deep_thought);
        try
        {
            f.get();
        }
        catch (vogon_exception const&)
        {
            std::cout << "Deep Thought Destroyed!\n";
        }

        // unsuccessful replicate_validate
        f = hpx::resiliency::experimental::async_replicate_validate(
            usr, &validate, &universal_answer);
        try
        {
            f.get();
        }
        catch (hpx::resiliency::experimental::abort_replicate_exception const&)
        {
            std::cout << "no universal answer!\n";
        }

        // aborted replicate
        f = hpx::resiliency::experimental::async_replicate(a, &no_answer);
        try
        {
            f.get();
        }
        catch (hpx::resiliency::experimental::abort_replicate_exception const&)
        {
            std::cout << "aborted universal answer calculation!\n";
        }

        // aborted replicate validate
        f = hpx::resiliency::experimental::async_replicate_validate(
            a, &validate, &no_answer);
        try
        {
            f.get();
        }
        catch (hpx::resiliency::experimental::abort_replicate_exception const&)
        {
            std::cout << "aborted universal answer calculation!\n";
        }

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

    desc_commandline.add_options()("sr-value",
        value<std::size_t>()->default_value(10),
        "Number of asynchronous function launches (curated for successful "
        "replicate example)");

    desc_commandline.add_options()("usr-value",
        value<std::size_t>()->default_value(6),
        "Number of asynchronous function launches (curated for unsuccessful "
        "replicate example)");

    desc_commandline.add_options()("a-value",
        value<std::size_t>()->default_value(1),
        "Number of asynchronous function launches (curated for aborted "
        "replicate example)");

    // Initialize and run HPX
    return hpx::init(desc_commandline, argc, argv);
}
