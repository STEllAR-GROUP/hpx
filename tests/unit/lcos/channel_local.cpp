//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/include/apply.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/atomic.hpp>

#include <numeric>
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
void sum(std::vector<int> const& s, hpx::lcos::local::channel<int> c)
{
    c.set(std::accumulate(s.begin(), s.end(), 0));      // send sum to channel
}

void calculate_sum()
{
    std::vector<int> s = { 7, 2, 8, -9, 4, 0 };
    hpx::lcos::local::channel<int> c;

    hpx::apply(&sum, std::vector<int>(s.begin(), s.begin() + s.size()/2), c);
    hpx::apply(&sum, std::vector<int>(s.begin() + s.size()/2, s.end()), c);

    int x = c.get(hpx::launch::sync);    // receive from c
    int y = c.get(hpx::launch::sync);

    int expected = std::accumulate(s.begin(), s.end(), 0);
    HPX_TEST_EQ(expected, x + y);
}

///////////////////////////////////////////////////////////////////////////////
void ping(
    hpx::lcos::local::send_channel<std::string> pings,
    std::string const& msg)
{
    pings.set(msg);
}

void pong(
    hpx::lcos::local::receive_channel<std::string> pings,
    hpx::lcos::local::send_channel<std::string> pongs)
{
    std::string msg = pings.get(hpx::launch::sync);
    pongs.set(msg);
}

void pingpong()
{
    hpx::lcos::local::channel<std::string> pings;
    hpx::lcos::local::channel<std::string> pongs;

    ping(pings, "passed message");
    pong(pings, pongs);

    std::string result = pongs.get(hpx::launch::sync);
    HPX_TEST_EQ(std::string("passed message"), result);
}

///////////////////////////////////////////////////////////////////////////////
void ping_void(hpx::lcos::local::send_channel<> pings)
{
    pings.set();
}

void pong_void(
    hpx::lcos::local::receive_channel<> pings,
    hpx::lcos::local::send_channel<> pongs,
    bool& pingponged)
{
    pings.get(hpx::launch::sync);
    pongs.set();
    pingponged = true;
}

void pingpong_void()
{
    hpx::lcos::local::channel<> pings;
    hpx::lcos::local::channel<> pongs;

    bool pingponged = false;

    ping_void(pings);
    pong_void(pings, pongs, pingponged);

    pongs.get(hpx::launch::sync);
    HPX_TEST(pingponged);
}

///////////////////////////////////////////////////////////////////////////////
void dispatch_work()
{
    hpx::lcos::local::channel<int> jobs;
    hpx::lcos::local::channel<> done;

    boost::atomic<int> received_jobs(0);
    boost::atomic<bool> was_closed(false);

    hpx::apply(
        [jobs, done, &received_jobs, &was_closed]() mutable
        {
            while(true)
            {
                auto next = jobs.get_checked();
                if (next.second)
                {
                    ++received_jobs;
                }
                else
                {
                    done.set();
                    was_closed = true;
                    break;
                }
            }
        });

    for (int j = 1; j <= 3; ++j)
    {
        jobs.set(j);
    }

    jobs.close();
    done.get(hpx::launch::sync);

    HPX_TEST_EQ(received_jobs.load(), 3);
    HPX_TEST(was_closed.load());
}

///////////////////////////////////////////////////////////////////////////////
void channel_range()
{
    boost::atomic<int> received_elements(0);

    hpx::lcos::local::channel<std::string> queue;
    queue.set("one");
    queue.set("two");
    queue.set("three");
    queue.close();

    for (auto const& elem : queue)
    {
        ++received_elements;
    }

    HPX_TEST_EQ(received_elements.load(), 3);
}

void channel_range_void()
{
    boost::atomic<int> received_elements(0);

    hpx::lcos::local::channel<> queue;
    queue.set();
    queue.set();
    queue.set();
    queue.close();

    for (auto const& elem : queue)
    {
        ++received_elements;
    }

    HPX_TEST_EQ(received_elements.load(), 3);
}

///////////////////////////////////////////////////////////////////////////////
void deadlock_test()
{
    bool caught_exception = false;
    try {
        hpx::lcos::local::channel<int> c;
        int value = c.get(hpx::launch::sync);
        HPX_TEST(false);
    }
    catch(hpx::exception const&) {
        caught_exception = true;
    }
    HPX_TEST(caught_exception);
}

void closed_channel_get()
{
    bool caught_exception = false;
    try {
        hpx::lcos::local::channel<int> c;
        c.close();

        int value = c.get(hpx::launch::sync);
        HPX_TEST(false);
    }
    catch(hpx::exception const&) {
        caught_exception = true;
    }
    HPX_TEST(caught_exception);
}

void closed_channel_get_generation()
{
    bool caught_exception = false;
    try {
        hpx::lcos::local::channel<int> c;
        c.set(42, 122);         // setting value for generation 122
        c.close();

        HPX_TEST_EQ(c.get(hpx::launch::sync, 122), 42);

        int value = c.get(hpx::launch::sync, 123); // asking for generation 123
        HPX_TEST(false);
    }
    catch(hpx::exception const&) {
        caught_exception = true;
    }
    HPX_TEST(caught_exception);
}

void closed_channel_set()
{
    bool caught_exception = false;
    try {
        hpx::lcos::local::channel<int> c;
        c.close();

        c.set(42);
        HPX_TEST(false);
    }
    catch(hpx::exception const&) {
        caught_exception = true;
    }
    HPX_TEST(caught_exception);
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    calculate_sum();
    pingpong();
    pingpong_void();
    dispatch_work();
    channel_range();
    channel_range_void();

    deadlock_test();
    closed_channel_get();
    closed_channel_get_generation();
    closed_channel_set();

    return hpx::util::report_errors();
}
