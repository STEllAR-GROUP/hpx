////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/hpx_main.hpp>
#include <hpx/apply.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/iostreams.hpp>

#include <functional>

struct data
{
    ///< For synchronizing two-phase initialization.
    hpx::lcos::local::event init;

    char const* msg;

    data() : init(), msg("uninitialized") {}

    void initialize(char const* p)
    {
        // We can only be called once.
        HPX_ASSERT(!init.occurred());
        msg = p;
        init.set();
    }
};

///////////////////////////////////////////////////////////////////////////////
void worker(std::size_t i, data& d, hpx::lcos::local::counting_semaphore& sem)
{
    d.init.wait();
    hpx::cout << d.msg << ": " << i << "\n" << hpx::flush;
    sem.signal();                   // signal main thread
}

///////////////////////////////////////////////////////////////////////////////
int main()
{
    data d;
    hpx::lcos::local::counting_semaphore sem;

    for (std::size_t i = 0; i < 10; ++i)
        hpx::apply(&worker, i, std::ref(d), std::ref(sem));

    d.initialize("initialized");    // signal the event

    // Wait for all threads to finish executing.
    sem.wait(10);

    return 0;
}

