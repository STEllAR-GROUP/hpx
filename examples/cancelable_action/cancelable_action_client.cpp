//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>

#include "cancelable_action/cancelable_action.hpp"

///////////////////////////////////////////////////////////////////////////////
void interrupt_do_it(examples::cancelable_action ca)
{
    // wait for one second before interrupting the (possibly remote) operation
    hpx::this_thread::sleep_for(boost::chrono::seconds(1));
    ca.cancel_it();
}

///////////////////////////////////////////////////////////////////////////////
void handle_interruption_using_exception(hpx::id_type const& id)
{
    // create a component encapsulating the 'do_it' operation
    examples::cancelable_action ca(id);

    // start a separate thread which will wait for a while and interrupt
    // the 'do_it' operation
    hpx::thread t(hpx::util::bind(interrupt_do_it, ca));

    try {
        // start some lengthy action, to be interrupted
        ca.do_it();
    }
    catch (hpx::thread_interrupted const&) {
        // we should get an error reporting hpx::thread_interrupted
        HPX_ASSERT(true);
    }

    // wait for the cancellation thread to exit
    t.join();
}

///////////////////////////////////////////////////////////////////////////////
void handle_interruption_using_error_code(hpx::id_type const& id)
{
    // create a component encapsulating the 'do_it' operation
    examples::cancelable_action ca(id);

    // start a separate thread which will wait for a while and interrupt
    // the 'do_it' operation
    hpx::thread t(hpx::util::bind(interrupt_do_it, ca));

    // start some lengthy action, to be interrupted
    hpx::error_code ec(hpx::lightweight);
    ca.do_it(ec);

    // we should get an error reporting hpx::thread_interrupted
    HPX_ASSERT(ec && ec.value() == hpx::thread_cancelled);

    // wait for the cancellation thread to exit
    t.join();
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    for (hpx::id_type const& id : hpx::find_all_localities())
    {
        handle_interruption_using_exception(id);
        handle_interruption_using_error_code(id);
    }
    return hpx::finalize();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Initialize and run HPX.
    return hpx::init(argc, argv);
}

