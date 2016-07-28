//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

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

int main(int argc, char* argv[])
{
    std::vector<std::string> cfg = { "hpx.os_threads=all" };
    HPX_TEST_EQ(hpx::init(argc, argv, cfg), 0);
    return hpx::util::report_errors();
}
