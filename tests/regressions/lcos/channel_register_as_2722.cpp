//  Copyright (c) 2017 Zach Byerly
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/util/lightweight_test.hpp>

HPX_REGISTER_CHANNEL(int);      // add to one source file

void send_values(hpx::lcos::channel<int> buffer)
{
    buffer.set(hpx::launch::sync, 42);
    buffer.set(hpx::launch::sync, 42);
}

void receive_values()
{
    hpx::lcos::channel<int> buffer;
    buffer.connect_to("my_channel");

    HPX_TEST_EQ(42, buffer.get(hpx::launch::sync));
    HPX_TEST_EQ(42, buffer.get(hpx::launch::sync));
}

int hpx_main(int argc, char **argv)
{
    {
        hpx::lcos::channel<int> buffer(hpx::find_here());
        buffer.register_as("my_channel");

        hpx::future<void> f1 = hpx::async(&send_values, buffer);
        hpx::future<void> f2 = hpx::async(&receive_values);

        hpx::wait_all(f1, f2);

    }   // unregisters 'buffer'

    return hpx::finalize();
}

int main(int argc, char **argv)
{
    HPX_TEST_EQ(0, hpx::init(argc, argv));
    return hpx::util::report_errors();
}
