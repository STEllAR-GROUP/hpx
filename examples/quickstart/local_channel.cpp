//  Copyright (c) 2016-2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This example demonstrates the use of a channel which is very similar to the
// equally named feature in the Go language.

#include <hpx/channel.hpp>
#include <hpx/future.hpp>
#include <hpx/init.hpp>

#include <cstddef>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
void sum(std::vector<int> const& s, hpx::lcos::local::channel<int> c)
{
    c.set(std::accumulate(s.begin(), s.end(), 0));    // send sum to channel
}

void calculate_sum()
{
    std::vector<int> s = {7, 2, 8, -9, 4, 0};
    hpx::lcos::local::channel<int> c;

    hpx::post(&sum,
        std::vector<int>(
            s.begin(), s.begin() + static_cast<std::ptrdiff_t>(s.size() / 2)),
        c);
    hpx::post(&sum,
        std::vector<int>(
            s.begin() + static_cast<std::ptrdiff_t>(s.size() / 2), s.end()),
        c);

    int x = c.get(hpx::launch::sync);    // receive from c
    int y = c.get(hpx::launch::sync);

    std::cout << "sum: " << x + y << std::endl;
}

///////////////////////////////////////////////////////////////////////////////
void ping(
    hpx::lcos::local::send_channel<std::string> pings, std::string const& msg)
{
    pings.set(msg);
}

void pong(hpx::lcos::local::receive_channel<std::string> pings,
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

    std::cout << "ping-ponged: " << pongs.get(hpx::launch::sync) << std::endl;
}

///////////////////////////////////////////////////////////////////////////////
void pingpong1()
{
    hpx::lcos::local::one_element_channel<std::string> pings;
    hpx::lcos::local::one_element_channel<std::string> pongs;

    for (int i = 0; i != 10; ++i)
    {
        ping(pings, "passed message");
        pong(pings, pongs);
        pongs.get(hpx::launch::sync);
    }

    std::cout << "ping-ponged 10 times\n";
}

void pingpong2()
{
    hpx::lcos::local::one_element_channel<std::string> pings;
    hpx::lcos::local::one_element_channel<std::string> pongs;

    ping(pings, "passed message");
    hpx::future<void> f1 = hpx::async([=]() { pong(pings, pongs); });

    ping(pings, "passed message");
    hpx::future<void> f2 = hpx::async([=]() { pong(pings, pongs); });

    pongs.get(hpx::launch::sync);
    pongs.get(hpx::launch::sync);

    f1.get();
    f2.get();

    std::cout << "ping-ponged with waiting\n";
}

///////////////////////////////////////////////////////////////////////////////
void dispatch_work()
{
    hpx::lcos::local::channel<int> jobs;
    hpx::lcos::local::channel<> done;

    hpx::post([jobs, done]() mutable {
        while (true)
        {
            hpx::error_code ec(hpx::throwmode::lightweight);
            int value = jobs.get(hpx::launch::sync, ec);
            if (!ec)
            {
                std::cout << "received job: " << value << std::endl;
            }
            else
            {
                std::cout << "received all jobs" << std::endl;
                done.set();
                break;
            }
        }
    });

    for (int j = 1; j <= 3; ++j)
    {
        jobs.set(j);
        std::cout << "sent job: " << j << std::endl;
    }

    jobs.close();
    std::cout << "sent all jobs" << std::endl;

    done.get(hpx::launch::sync);
}

///////////////////////////////////////////////////////////////////////////////
void channel_range()
{
    hpx::lcos::local::channel<std::string> queue;

    queue.set("one");
    queue.set("two");
    queue.close();

    for (auto const& elem : queue)
        std::cout << elem << std::endl;
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    calculate_sum();
    pingpong();
    pingpong1();
    pingpong2();
    dispatch_work();
    channel_range();

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    return hpx::local::init(hpx_main, argc, argv);
}
