//  Copyright (c) 2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Verify that #1550 was properly fixed (Properly fix HPX_DEFINE_*_ACTION macros)

#include <hpx/hpx_init.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/async.hpp>
#include <hpx/util/lightweight_test.hpp>

bool called_test_action = false;

namespace mynamespace
{
    void test()
    {
        called_test_action = true;
    }

    HPX_DEFINE_PLAIN_ACTION(test);
}

typedef mynamespace::test_action mynamespace_test_action;

HPX_REGISTER_ACTION(mynamespace_test_action);

int hpx_main(int argc, char* argv[])
{
    {
        typedef mynamespace_test_action func;
        hpx::async<func>(hpx::find_here()).get();
    }

    HPX_TEST(called_test_action);

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ(hpx::init(argc, argv), 0);
    return hpx::util::report_errors();
}
