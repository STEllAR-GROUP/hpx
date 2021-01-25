////////////////////////////////////////////////////////////////////////////////
//  Copyright (C) 2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
////////////////////////////////////////////////////////////////////////////////

#include <hpx/config.hpp>
#include <hpx/allocator_support/aligned_allocator.hpp>
#include <hpx/functional/bind.hpp>
#include <hpx/modules/testing.hpp>

#include <hpx/modules/program_options.hpp>
#include <boost/lockfree/queue.hpp>

#include <cstdint>
#include <iostream>
#include <thread>
#include <vector>

using queue = boost::lockfree::queue<std::uint64_t,
    hpx::util::aligned_allocator<std::uint64_t>>;

std::vector<queue*> queues;
std::vector<std::uint64_t> stolen;

std::uint64_t threads = 2;
std::uint64_t items = 500000;

bool get_next_thread(std::uint64_t num_thread)
{
    std::uint64_t r = 0;

    if ((*queues[num_thread]).pop(r))
        return true;

    for (std::uint64_t i = 0; i < threads; ++i)
    {
        if (i == num_thread)
            continue;

        if ((*queues[i]).pop(r))
        {
            ++stolen[num_thread];
            return true;
        }
    }

    return false;
}

void worker_thread(std::uint64_t num_thread)
{
    //    while (get_next_thread(num_thread))
    //        {}
    for (std::uint64_t i = 0; i < items; ++i)
    {
        bool result = get_next_thread(num_thread);
        HPX_TEST(result);
    }
}

int main(int argc, char** argv)
{
    using hpx::program_options::command_line_parser;
    using hpx::program_options::notify;
    using hpx::program_options::options_description;
    using hpx::program_options::store;
    using hpx::program_options::value;
    using hpx::program_options::variables_map;

    variables_map vm;

    options_description desc_cmdline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    // clang-format off
    desc_cmdline.add_options()
        ("help,h", "print out program usage (this message)")
        ("threads,t", value<std::uint64_t>(&threads)->default_value(2),
         "the number of worker threads inserting objects into the fifo")
        ("items,i", value<std::uint64_t>(&items)->default_value(500000),
         "the number of items to create per queue")
    ;
    // clang-format on

    store(command_line_parser(argc, argv)
              .options(desc_cmdline)
              .allow_unregistered()
              .run(),
        vm);

    notify(vm);

    // print help screen
    if (vm.count("help"))
    {
        std::cout << desc_cmdline;
        return hpx::util::report_errors();
    }

    if (vm.count("threads"))
        threads = vm["threads"].as<std::uint64_t>();

    stolen.resize(threads);

    for (std::uint64_t i = 0; i < threads; ++i)
    {
        queues.push_back(new queue(items));

        for (std::uint64_t j = 0; j < items; ++j)
            (*queues[i]).push(j);

        HPX_TEST(!(*queues[i]).empty());
    }

    {
        std::vector<std::thread> tg;

        for (std::uint64_t i = 0; i != threads; ++i)
            tg.push_back(std::thread(hpx::util::bind(&worker_thread, i)));

        for (std::thread& t : tg)
        {
            if (t.joinable())
                t.join();
        }
    }

    for (std::uint64_t i = 0; i < threads; ++i)
        HPX_TEST_EQ(stolen[i], std::uint64_t(0));

    for (std::uint64_t i = 0; i < threads; ++i)
        delete queues[i];

    return hpx::util::report_errors();
}
