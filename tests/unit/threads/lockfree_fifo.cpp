////////////////////////////////////////////////////////////////////////////////
//  Copyright (C) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
////////////////////////////////////////////////////////////////////////////////

#include <hpx/config.hpp>

#include <boost/thread/thread.hpp>
#include <boost/program_options.hpp>

#include <boost/detail/lightweight_test.hpp>

#include <boost/lockfree/queue.hpp>

std::vector<boost::lockfree::queue<boost::uint64_t>*> queues;
std::vector<boost::uint64_t> stolen;

boost::uint64_t threads = 2;
boost::uint64_t items = 500000;

bool get_next_thread(boost::uint64_t num_thread)
{
    boost::uint64_t r = 0;

    if ((*queues[num_thread]).pop(r))
        return true;

    for (boost::uint64_t i = 0; i < threads; ++i)
    {
        if (i == num_thread) continue;

        if ((*queues[i]).pop(r))
        {
            ++stolen[num_thread];
            return true;
        }
    }

    return false;
}

void worker_thread(boost::uint64_t num_thread)
{
//    while (get_next_thread(num_thread))
//        {}
    for (boost::uint64_t i = 0; i < items; ++i)
    {
        bool result = get_next_thread(num_thread);
        BOOST_TEST(result);
    }
}

int main(int argc, char** argv)
{
    using boost::program_options::variables_map;
    using boost::program_options::options_description;
    using boost::program_options::value;
    using boost::program_options::store;
    using boost::program_options::command_line_parser;
    using boost::program_options::notify;

    variables_map vm;

    options_description
        desc_cmdline("Usage: " HPX_APPLICATION_STRING " [options]");

    desc_cmdline.add_options()
        ("help,h", "print out program usage (this message)")
        ("threads,t", value<boost::uint64_t>(&threads)->default_value(2),
         "the number of worker threads inserting objects into the fifo")
        ("items,i", value<boost::uint64_t>(&items)->default_value(500000),
         "the number of items to create per queue")
    ;

    store(
        command_line_parser(argc,
            argv).options(desc_cmdline).allow_unregistered().run(),vm);

    notify(vm);

    // print help screen
    if (vm.count("help"))
    {
        std::cout << desc_cmdline;
        return boost::report_errors();
    }

    if (vm.count("threads"))
        threads = vm["threads"].as<boost::uint64_t>();

    stolen.resize(threads);

    for (boost::uint64_t i = 0; i < threads; ++i)
    {
        queues.push_back(new boost::lockfree::queue<boost::uint64_t>(items));

        for (boost::uint64_t j = 0; j < items; ++j)
            (*queues[i]).push(j);

        BOOST_TEST(!(*queues[i]).empty());
    }

    {
        boost::thread_group tg;

        for (boost::uint64_t i = 0; i != threads; ++i)
            tg.create_thread(boost::bind(&worker_thread, i));

        tg.join_all();
    }

    for (boost::uint64_t i = 0; i < threads; ++i)
        BOOST_TEST(stolen[i] == 0);

    for (boost::uint64_t i = 0; i < threads; ++i)
        delete queues[i];

    return boost::report_errors();
}

