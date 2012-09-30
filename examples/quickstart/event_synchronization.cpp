////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011-2012 Bryce Adelstein-Lelbach 
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/hpx_main.hpp>
#include <hpx/apply.hpp>
#include <hpx/lcos/local/event_semaphore.hpp>
#include <hpx/include/iostreams.hpp>

struct data
{
    ///< For synchronizing two-phase initialization.
    hpx::lcos::local::event_semaphore init;

    char const* msg;

    data() : init(), msg("uninitialized") {}

    void initialize(char const* p)
    {
        // We can only be called once.
        BOOST_ASSERT(!init.occurred());
        msg = p;
        init.set();
    }
};

///////////////////////////////////////////////////////////////////////////////
void worker(data* d)
{
    hpx::cout << d->msg << hpx::flush;
}

///////////////////////////////////////////////////////////////////////////////
int main()
{
    data d;

    for (std::size_t i = 0; i < 10; ++i)
        hpx::apply(&worker, &d);

    d.initialize("initialized");

    // Wait for all threads to finish executing.
    do {
        hpx::this_thread::suspend();
    } while (hpx::threads::get_thread_count() > 1);

    return 0;
}

