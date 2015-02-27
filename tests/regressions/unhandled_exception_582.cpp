//  Copyright (c) 2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Checking that #582 was fixed

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/util/lightweight_test.hpp>

int hpx_main(int argc, char ** argv)
{
    HPX_THROW_EXCEPTION(hpx::invalid_status, "hpx_main", "testing");
    return hpx::finalize();
}

int main(int argc, char **argv)
{
    bool caught_exception = false;
    try {
        hpx::init(argc, argv);
    }
    catch(hpx::exception const&) {
        caught_exception = true;
    }
    catch(...) {
        HPX_TEST(false);
    }
    HPX_TEST(caught_exception);

    return hpx::util::report_errors();
}
