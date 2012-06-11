//  Copyright (c) 2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>

#include <boost/detail/lightweight_test.hpp>

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
    hpx::lcos::wait(hpx::async<throw_exception_action>(hpx::find_here()));

    hpx::finalize();
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options.
    options_description cmdline("usage: " HPX_APPLICATION_STRING " [options]");

    // Initialize and run HPX.
    return hpx::init(cmdline, argc, argv);
}

