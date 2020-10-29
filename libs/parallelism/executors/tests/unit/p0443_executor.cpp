//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/execution.hpp>
#include <hpx/functional.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>

///////////////////////////////////////////////////////////////////////////////
void test(int passed_through, hpx::thread::id parent_id)
{
    HPX_TEST_EQ(passed_through, 42);
    HPX_TEST_NEQ(hpx::this_thread::get_id(), parent_id);
}

void test_executor()
{
    hpx::execution::experimental::p0443_executor exec{};
    hpx::execution::experimental::execute(
        exec, hpx::util::bind(test, 42, hpx::this_thread::get_id()));
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(int argc, char* argv[])
{
    test_executor();

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ_MSG(
        hpx::init(argc, argv), 0, "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
