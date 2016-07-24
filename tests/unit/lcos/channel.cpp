//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/include/apply.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/atomic.hpp>

#include <numeric>
#include <string>
#include <vector>

typedef std::string string_type;

HPX_REGISTER_CHANNEL(int);
HPX_REGISTER_CHANNEL(string_type);
HPX_REGISTER_CHANNEL(void);

///////////////////////////////////////////////////////////////////////////////
void sum(std::vector<int> const& s, hpx::lcos::channel<int> c)
{
    c.set(std::accumulate(s.begin(), s.end(), 0));      // send sum to channel
}
HPX_PLAIN_ACTION(sum);

void calculate_sum(hpx::id_type const& loc)
{
    std::vector<int> s = { 7, 2, 8, -9, 4, 0 };
    hpx::lcos::channel<int> c (loc);

    hpx::apply(sum_action(), loc,
        std::vector<int>(s.begin(), s.begin() + s.size()/2), c);
    hpx::apply(sum_action(), loc,
        std::vector<int>(s.begin() + s.size()/2, s.end()), c);

    int x = c.get();    // receive from c
    int y = c.get();

    int expected = std::accumulate(s.begin(), s.end(), 0);
    HPX_TEST_EQ(expected, x + y);
}

///////////////////////////////////////////////////////////////////////////////
void ping(
    hpx::lcos::send_channel<std::string> pings,
    std::string const& msg)
{
    pings.set(msg);
}

void pong(
    hpx::lcos::receive_channel<std::string> pings,
    hpx::lcos::send_channel<std::string> pongs)
{
    std::string msg = pings.get();
    pongs.set(msg);
}

void pingpong(hpx::id_type const& loc)
{
    hpx::lcos::channel<std::string> pings(loc);
    hpx::lcos::channel<std::string> pongs(loc);

    ping(pings, "passed message");
    pong(pings, pongs);

    std::string result = pongs.get();
    HPX_TEST_EQ(std::string("passed message"), result);
}

///////////////////////////////////////////////////////////////////////////////
void ping_void(hpx::lcos::send_channel<> pings)
{
    pings.set();
}

void pong_void(
    hpx::lcos::receive_channel<> pings,
    hpx::lcos::send_channel<> pongs,
    bool& pingponged)
{
    pings.get();
    pongs.set();
    pingponged = true;
}

void pingpong_void(hpx::id_type const& loc)
{
    hpx::lcos::channel<> pings(loc);
    hpx::lcos::channel<> pongs(loc);

    bool pingponged = false;

    ping_void(pings);
    pong_void(pings, pongs, pingponged);

    pongs.get();
    HPX_TEST(pingponged);
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    hpx::id_type here = hpx::find_here();

    calculate_sum(here);
    pingpong(here);
    pingpong_void(here);
//     dispatch_work();
//     channel_range();
//     channel_range_void();
//
//     deadlock_test();
//     closed_channel_get();
//     closed_channel_get_generation();
//     closed_channel_set();

    return hpx::util::report_errors();
}
