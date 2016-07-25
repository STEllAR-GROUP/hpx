//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This example demonstrates the use a a channel which is very similar to the
// equally named feature in the Go language.

#include <hpx/hpx_main.hpp>
#include <hpx/include/apply.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/include/lcos.hpp>

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

    int x = c.get_sync();    // receive from c
    int y = c.get_sync();

    hpx::cout << "sum: " << x + y << std::endl;
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
    std::string msg = pings.get_sync();
    pongs.set(msg);
}

void pingpong()
{
    hpx::lcos::local::channel<std::string> pings;
    hpx::lcos::local::channel<std::string> pongs;

    ping(pings, "passed message");
    pong(pings, pongs);

    hpx::cout << "ping-ponged: " << pongs.get_sync() << std::endl;
}

///////////////////////////////////////////////////////////////////////////////
void dispatch_work()
{
    hpx::lcos::local::channel<int> jobs;
    hpx::lcos::local::channel<> done;

    hpx::apply(
        [jobs, done]() mutable
        {
            while(true)
            {
                auto next = jobs.get_checked();
                if (next.second)
                {
                    hpx::cout << "received job: " << next.first << std::endl;
                }
                else
                {
                    hpx::cout << "received all jobs" << std::endl;
                    done.set();
                    break;
                }
            }
        });

    for (int j = 1; j <= 3; ++j)
    {
        jobs.set(j);
        hpx::cout << "sent job: " << j << std::endl;
    }

    jobs.close();
    hpx::cout << "sent all jobs" << std::endl;

    done.get_sync();
}

///////////////////////////////////////////////////////////////////////////////
void channel_range()
{
    hpx::lcos::local::channel<std::string> queue;

    queue.set("one");
    queue.set("two");
    queue.close();

    for (auto const& elem : queue)
        hpx::cout << elem << std::endl;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    calculate_sum();
    pingpong();
    dispatch_work();
    channel_range();

    return 0;
}
