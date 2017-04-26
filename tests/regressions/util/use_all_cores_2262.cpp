//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test verifies that 'hpx.os_threads=all' is equivalent to specifying
// all of the avalable cores (see #2262).

#include <hpx/hpx_init.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <string>
#include <vector>

int hpx_main(int argc, char* argv[])
{
    HPX_TEST_EQ(
        hpx::threads::hardware_concurrency(),
        hpx::get_os_thread_count());

    return hpx::finalize();
}

// Ignore all command line options to avoid any interference with the test
// runners.
char* argv[] =
{
    (char*)"use_all_cores_2262", nullptr
};

int main()
{
    std::vector<std::string> cfg = { "hpx.os_threads=all" };
    HPX_TEST_EQ(hpx::init(1, argv, cfg), 0);
    return hpx::util::report_errors();
}
