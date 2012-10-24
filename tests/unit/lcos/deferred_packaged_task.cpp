//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/thread.hpp>
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

typedef hpx::lcos::deferred_packaged_task<null_action> null_future;

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map&)
{
    {
        // create an explicit future
        null_thread_executed = false;
        {
            null_future f(hpx::find_here());
            HPX_TEST(f.get());
        }
        HPX_TEST(null_thread_executed);
    }

    hpx::finalize();       // Initiate shutdown of the runtime system.
    return hpx::util::report_errors();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options.
    options_description cmdline("usage: " HPX_APPLICATION_STRING " [options]");

    // We force this test to use several threads by default.
    using namespace boost::assign;
    std::vector<std::string> cfg;
    cfg += "hpx.os_threads=" +
        boost::lexical_cast<std::string>(hpx::threads::hardware_concurrency());

    // Initialize and run HPX
    return hpx::init(cmdline, argc, argv, cfg);
}

