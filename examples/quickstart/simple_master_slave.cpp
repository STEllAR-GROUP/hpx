//  Copyright (c) 2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// The purpose of this example is to demonstrate how HPX actions can be used to
// build a simple master-slave application. The master (locality 0) assigns work
// to the slaves (all other localities). Note that if this application is run on
// one locality only it uses the same locality for the master and the slave
// functionalities.
//
// The slaves receive a message that encodes how many sub-tasks of a certain
// type they should spawn locally.

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include <iostream>
#include <random>
#include <vector>

// Below are the three different tasks a slave can execute
enum class task_type
{
    one = 1,
    two = 2,
    three = 3
};

// task_type::one
void work_item1(int sequence_number)
{
    std::cout << hpx::util::format("locality {}: work_item1: {}\n",
        hpx::get_locality_id(), sequence_number);
}

// task_type::two
void work_item2(int sequence_number)
{
    std::cout << hpx::util::format("locality {}: work_item2: {}\n",
        hpx::get_locality_id(), sequence_number);
}

// task_type::three
void work_item3(int sequence_number)
{
    std::cout << hpx::util::format("locality {}: work_item3: {}\n",
        hpx::get_locality_id(), sequence_number);
}

bool slave_operation(int count, task_type t)
{
    bool result = true;
    std::vector<hpx::future<void>> tasks;
    tasks.reserve(count);

    for (int i = 0; i != count; ++i)
    {
        switch (t)
        {
        case task_type::one:
            tasks.push_back(hpx::async(work_item1, i));
            break;

        case task_type::two:
            tasks.push_back(hpx::async(work_item2, i));
            break;

        case task_type::three:
            tasks.push_back(hpx::async(work_item3, i));
            break;

        default:
            std::cerr << hpx::util::format(
                "locality {}: unknown task type: {}\n", hpx::get_locality_id(),
                int(t));
            result = false;
            break;
        }
    }

    hpx::wait_all(std::move(tasks));
    return result;
}
HPX_PLAIN_ACTION(slave_operation)

int hpx_main(hpx::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int) std::random_device{}();
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::mt19937 gen(seed);

    std::uniform_int_distribution<> repeat_dist(1, 3);
    std::uniform_int_distribution<> count_dist(1, 10);
    std::uniform_int_distribution<> type_dist(1, 3);

    // Submit work locally as well if there is just one locality
    std::vector<hpx::id_type> slave_localities = hpx::find_all_localities();
    if (slave_localities.size() > 1)
    {
        // submit work only remotely otherwise
        slave_localities = hpx::find_remote_localities();
    }

    // schedule random amount of slave tasks to each slave locality
    std::vector<hpx::future<bool>> slave_tasks;

    auto repeat = repeat_dist(gen);
    for (auto const& locality : slave_localities)
    {
        for (int i = 0; i != repeat; ++i)
        {
            auto count = count_dist(gen);
            auto type = static_cast<task_type>(type_dist(gen));

            slave_tasks.push_back(
                hpx::async(slave_operation_action(), locality, count, type));
        }
    }

    hpx::wait_all(slave_tasks);

    for (auto&& f : slave_tasks)
    {
        if (!f.get())
        {
            std::cerr << "One of the tasks failed!\n";
            break;
        }
    }

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // define command line options
    hpx::program_options::options_description desc_commandline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()("seed,s",
        hpx::program_options::value<unsigned int>(),
        "the random number generator seed to use for this run");

    // Initialize and run HPX
    hpx::init_params init_args;
    init_args.desc_cmdline = desc_commandline;

    return hpx::init(argc, argv, init_args);
}
