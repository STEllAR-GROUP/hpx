////////////////////////////////////////////////////////////////////////////////
//  Copyright (C) 2009-2011 Tim Blechmann
//  Copyright (C)      2011 Bryce Lelbach 
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <deque>
#include <queue>

#include <boost/atomic.hpp>
#include <boost/thread/thread.hpp>
#include <boost/program_options.hpp>

#include <hpx/util/lightweight_test.hpp>
#include <hpx/util/spinlock_pool.hpp>

boost::atomic<std::size_t> producer_count(0);
boost::atomic<std::size_t> consumer_count(0);

struct test_tag {};
typedef hpx::util::spinlock_pool<test_tag> test_mutex_type;

struct queue_type
{
    void enqueue(std::size_t value)
    {
        test_mutex_type::scoped_lock l(this);
        data.push(value);
    }

    bool dequeue(std::size_t& return_)
    {
        test_mutex_type::scoped_lock l(this);
        if (data.empty())
            return false;
        else
        {
            return_ = data.front();
            data.pop();
            return true;
        }
    }

    std::queue<std::size_t> data;
} queue;

std::size_t iterations = 100000;
std::size_t producer_thread_count = 4;
std::size_t consumer_thread_count = 4;

volatile bool done = false;

void producer()
{
    for (std::size_t i = 0; i != iterations; ++i)
    {
        std::size_t value = ++producer_count;
        queue.enqueue(value);
    }
}

void consumer()
{
    std::size_t value(0);

    while (!done)
    {
        while (queue.dequeue(value))
            ++consumer_count;
    }

    while (queue.dequeue(value))
        ++consumer_count;
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
        ("producer-threads,p", value<std::size_t>(), 
         "the number of worker threads inserting objects into the queue "
         "(default: 4)") 
        ("consumer-threads,c", value<std::size_t>(), 
         "the number of worker threads removing objects into the queue "
         "(default: 4)") 
        ("iterations,i", value<std::size_t>(), 
         "the number of iterations (default: 100000)") 
    ;

    store(command_line_parser(argc, argv).options(desc_cmdline).run(), vm);

    notify(vm);

    // print help screen
    if (vm.count("help"))
    {
        std::cout << desc_cmdline;
        return hpx::util::report_errors();
    }

    if (vm.count("consumer-threads"))
        consumer_thread_count = vm["consumer-threads"].as<std::size_t>();
    
    if (vm.count("producer-threads"))
        producer_thread_count = vm["producer-threads"].as<std::size_t>();
    
    if (vm.count("iterations"))
        iterations = vm["iterations"].as<std::size_t>();

    boost::thread_group producer_threads, consumer_threads;

    for (std::size_t i = 0; i != producer_thread_count; ++i)
        producer_threads.create_thread(producer);

    for (std::size_t i = 0; i != consumer_thread_count; ++i)
        consumer_threads.create_thread(consumer);

    producer_threads.join_all();
    done = true;

    consumer_threads.join_all();

    HPX_TEST_EQ(producer_count, consumer_count);

    std::cout << "produced " << producer_count << " objects.\n"
              << "consumed " << consumer_count << " objects.\n";

    return hpx::util::report_errors();
}

