//  Copyright (c) 2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/hpx.hpp>
#include <hpx/util/lightweight_test.hpp>

///////////////////////////////////////////////////////////////////////////////
bool thrown_exception = false;

void throw_hpx_exception()
{
    thrown_exception = true;
    HPX_THROW_EXCEPTION(hpx::bad_request,
        "throw_hpx_exception", "testing HPX exception");
}

HPX_PLAIN_ACTION(throw_hpx_exception, throw_hpx_exception_action);

///////////////////////////////////////////////////////////////////////////////
int main()
{
    bool caught_exception = false;
    thrown_exception = false;

    try {
        throw_hpx_exception_action act;
        act(hpx::find_here());
    }
    catch (hpx::exception const&) {
        caught_exception = true;
    }

    HPX_TEST(thrown_exception);
    HPX_TEST(caught_exception);

    return hpx::util::report_errors();
}
