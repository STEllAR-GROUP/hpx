////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/assert.hpp>
#include <hpx/future.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/synchronization.hpp>

#include <cstddef>
#include <functional>
#include <iostream>

struct data
{
    ///< For synchronizing two-phase initialization.
    hpx::lcos::local::event init;

    char const* msg;

    data()
      : init()
      , msg("uninitialized")
    {
    }

    void initialize(char const* p)
    {
        // We can only be called once.
        HPX_ASSERT(!init.occurred());
        msg = p;
        init.set();
    }
};

///////////////////////////////////////////////////////////////////////////////
void worker(std::size_t i, data& d, hpx::counting_semaphore_var<>& sem)
{
    d.init.wait();
    std::cout << d.msg << ": " << i << "\n" << std::flush;
    sem.signal();    // signal main thread
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    data d;
    hpx::counting_semaphore_var<> sem;

    for (std::size_t i = 0; i < 10; ++i)
        hpx::post(&worker, i, std::ref(d), std::ref(sem));

    d.initialize("initialized");    // signal the event

    // Wait for all threads to finish executing.
    sem.wait(10);

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    return hpx::local::init(hpx_main, argc, argv);
}
