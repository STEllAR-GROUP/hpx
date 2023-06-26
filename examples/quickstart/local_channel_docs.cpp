//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This example is meant for inclusion in the documentation.

#include <hpx/assert.hpp>
#include <hpx/future.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/synchronization.hpp>

#include <iostream>
#include <numeric>
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
void minimal_channel()
{
    //[local_channel_minimal
    hpx::lcos::local::channel<int> c;
    hpx::future<int> f = c.get();
    HPX_ASSERT(!f.is_ready());
    c.set(42);
    HPX_ASSERT(f.is_ready());
    std::cout << f.get() << std::endl;
    //]
}

///////////////////////////////////////////////////////////////////////////////
//[local_channel_send_receive
void do_something(hpx::lcos::local::receive_channel<int> c,
    hpx::lcos::local::send_channel<> done)
{
    // prints 43
    std::cout << c.get(hpx::launch::sync) << std::endl;
    // signal back
    done.set();
}

void send_receive_channel()
{
    hpx::lcos::local::channel<int> c;
    hpx::lcos::local::channel<> done;

    hpx::post(&do_something, c, done);

    // send some value
    c.set(43);
    // wait for thread to be done
    done.get().wait();
}
//]

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    minimal_channel();
    send_receive_channel();

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    return hpx::local::init(hpx_main, argc, argv);
}
