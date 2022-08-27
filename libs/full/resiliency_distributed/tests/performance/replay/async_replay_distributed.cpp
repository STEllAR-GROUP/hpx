//  Copyright (c) 2019 National Technology & Engineering Solutions of Sandia,
//                     LLC (NTESS).
//  Copyright (c) 2018-2022 Hartmut Kaiser
//  Copyright (c) 2018-2019 Adrian Serio
//  Copyright (c) 2019-2020 Nikunj Gupta
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)

#include <hpx/actions_base/plain_action.hpp>
#include <hpx/assert.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/runtime.hpp>
#include <hpx/modules/futures.hpp>
#include <hpx/modules/resiliency.hpp>
#include <hpx/modules/resiliency_distributed.hpp>
#include <hpx/modules/testing.hpp>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <ctime>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

int universal_ans(
    std::vector<hpx::id_type> f_locales, std::size_t err, std::size_t size)
{
    std::vector<hpx::future<int>> local_tasks;

    for (std::size_t i = 0; i < 10; ++i)
    {
        local_tasks.push_back(hpx::async([size]() {
            // Pretending to do some useful work
            std::size_t start = hpx::chrono::high_resolution_clock::now();

            while ((hpx::chrono::high_resolution_clock::now() - start) <
                (size * 100))
            {
            }

            return 42;
        }));
    }

    hpx::wait_all(local_tasks);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<std::size_t> dist(1, 100);

    bool is_faulty = false;

    // Check if the node is faulty
    for (const auto& locale : f_locales)
    {
        // Throw a runtime error in case the node is faulty
        if (locale == hpx::find_here())
        {
            is_faulty = true;
            if (dist(gen) < err * 10)
                throw std::runtime_error("runtime error occurred.");
        }
    }

    if (!is_faulty && dist(gen) < err)
        throw std::runtime_error("runtime error occurred.");

    return 42;
}

HPX_PLAIN_ACTION(universal_ans, universal_action)

bool validate(int ans)
{
    return ans == 42;
}

int hpx_main(hpx::program_options::variables_map& vm)
{
    std::size_t f_nodes = vm["f-nodes"].as<std::size_t>();
    std::size_t err = vm["error"].as<std::size_t>();
    std::size_t size = vm["size"].as<std::size_t>();
    std::size_t num_tasks_total = vm["num-tasks"].as<std::size_t>();

    universal_action ac;
    std::vector<hpx::id_type> locales = hpx::find_all_localities();
    std::size_t num_localities = hpx::get_num_localities(hpx::launch::sync);
    std::size_t num_tasks = num_tasks_total / num_localities;

    std::size_t rank = hpx::get_locality_id();
    if (rank == 0)
        num_tasks += (num_tasks_total % num_localities);

    // Make sure that the number of faulty nodes are less than the number of
    // localities we work on.
    HPX_ASSERT(f_nodes < locales.size());

    // List of faulty nodes
    std::vector<hpx::id_type> f_locales;
    std::vector<std::size_t> visited;

    // Mark nodes as faulty
    for (std::size_t i = 0; i < f_nodes; ++i)
    {
        std::size_t num = std::rand() % locales.size();
        while (visited.end() != std::find(visited.begin(), visited.end(), num))
        {
            num = std::rand() % locales.size();
        }

        f_locales.push_back(locales.at(num));
    }

    {
        hpx::chrono::high_resolution_timer t;

        std::vector<hpx::future<int>> tasks;
        for (std::size_t i = 0; i < num_tasks; ++i)
        {
            tasks.push_back(hpx::resiliency::experimental::async_replay(
                locales, ac, f_locales, err, size));

            std::rotate(locales.begin(), locales.begin() + 1, locales.end());
        }

        HPX_TEST(!hpx::wait_all_nothrow(tasks));

        double elapsed = t.elapsed();

        if (rank == 0)
            std::cout << "Replay: " << elapsed << std::endl;
    }

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    namespace po = hpx::program_options;

    // Configure application-specific options
    po::options_description desc_commandline;

    // clang-format off
    desc_commandline.add_options()
        ("f-nodes", po::value<std::size_t>()->default_value(1),
            "Number of faulty nodes to be injected")
        ("error", po::value<std::size_t>()->default_value(5),
            "Error rates for all nodes. Faulty nodes will have 10x error rates.")
        ("size", po::value<std::size_t>()->default_value(200),
            "Grain size of a task")
        ("num-tasks", po::value<std::size_t>()->default_value(100),
            "Number of tasks to invoke")
    ;
    // clang-format on

    // Initialize and run HPX, this example requires to run hpx_main on all
    // localities
    std::vector<std::string> cfg = {
        "hpx.run_hpx_main!=1",
    };

    // Initialize and run HPX
    hpx::init_params params;
    params.desc_cmdline = desc_commandline;
    params.cfg = std::move(cfg);
    return hpx::init(argc, argv, params);
}

#endif
