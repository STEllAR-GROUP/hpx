//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/components.hpp>
#include <hpx/util/lightweight_test.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;

///////////////////////////////////////////////////////////////////////////////
bool null_thread_executed = false;

bool null_thread()
{
    HPX_TEST(!null_thread_executed);
    null_thread_executed = true;
    return true;
}

// Define the boilerplate code necessary for the function 'null_thread'
// to be invoked as an HPX action (by a HPX future)
typedef hpx::actions::plain_result_action0<bool, null_thread> null_action;

HPX_REGISTER_PLAIN_ACTION(null_action);

///////////////////////////////////////////////////////////////////////////////
int int_thread()
{
    return 9000;
}

// Define the boilerplate code necessary for the function 'int_thread'
// to be invoked as an HPX action (by a HPX future)
typedef hpx::actions::plain_result_action0<int, int_thread> int_action;

HPX_REGISTER_PLAIN_ACTION(int_action);

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map&)
{
    using hpx::lcos::async;

    // create an explicit future
    null_thread_executed = false;
    {
        HPX_TEST(async<null_action>(hpx::find_here()).get());
    }
    HPX_TEST(null_thread_executed);

    // create an implicit future
    null_thread_executed = false;
    {
        HPX_TEST(hpx::lcos::wait(async<null_action>(hpx::find_here())));
    }
    HPX_TEST(null_thread_executed);

    //test two successive 'get' from a promise
    hpx::lcos::future<int> int_promise(async<int_action>(hpx::find_here()));
    HPX_TEST(int_promise.get() == int_promise.get());

    hpx::finalize();       // Initiate shutdown of the runtime system.
    return hpx::util::report_errors();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options.
    options_description
        desc_commandline("usage: " HPX_APPLICATION_STRING " [options]");

    // Initialize and run HPX.
    return hpx::init(desc_commandline, argc, argv);
}

