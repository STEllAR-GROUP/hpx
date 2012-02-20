//  Copyright (c) 2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/util/lightweight_test.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;

///////////////////////////////////////////////////////////////////////////////
bool thrown_exception = false;

void throw_hpx_exception()
{
    thrown_exception = true;
    HPX_THROW_EXCEPTION(hpx::bad_request,
        "throw_hpx_exception", "testing HPX exception");
}

typedef hpx::actions::plain_action0<throw_hpx_exception> throw_exception_action;

HPX_REGISTER_PLAIN_ACTION(throw_exception_action);

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm)
{
    bool caught_exception = false;
    thrown_exception = false;

    try {
        hpx::lcos::wait(hpx::lcos::async<throw_exception_action>(hpx::find_here()));
    }
    catch (hpx::exception const&) {
        caught_exception = true;
    }

    HPX_TEST(thrown_exception);
    HPX_TEST(caught_exception);

    hpx::finalize();
    return hpx::util::report_errors();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options.
    options_description cmdline("usage: " HPX_APPLICATION_STRING " [options]");

    // Initialize and run HPX.
    return hpx::init(cmdline, argc, argv);
}

