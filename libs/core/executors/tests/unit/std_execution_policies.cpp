//  Copyright (c) 2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/execution.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>

///////////////////////////////////////////////////////////////////////////////
void static_checks()
{
    static_assert(
        hpx::is_execution_policy<std::execution::sequenced_policy>::value,
        "hpx::is_execution_policy<std::execution::sequenced_policy>::value");
    static_assert(
        hpx::is_execution_policy<std::execution::parallel_policy>::value,
        "hpx::is_execution_policy<std::execution::parallel_policy>::value");
    static_assert(hpx::is_execution_policy<
                      std::execution::parallel_unsequenced_policy>::value,
        "hpx::is_execution_policy<std::execution::parallel_unsequenced_policy>:"
        ":value");

    static_assert(hpx::is_sequenced_execution_policy<
                      std::execution::sequenced_policy>::value,
        "hpx::is_sequenced_execution_policy<std::execution::sequenced_policy>::"
        "value");
    static_assert(!hpx::is_sequenced_execution_policy<
                      std::execution::parallel_policy>::value,
        "!hpx::is_sequenced_execution_policy<std::execution::parallel_policy>::"
        "value");
    static_assert(!hpx::is_sequenced_execution_policy<
                      std::execution::parallel_unsequenced_policy>::value,
        "!hpx::is_sequenced_execution_policy<std::execution::parallel_"
        "unsequenced_policy>::value");

    static_assert(!hpx::is_parallel_execution_policy<
                      std::execution::sequenced_policy>::value,
        "!hpx::is_sequenced_execution_policy<std::execution::sequenced_policy>:"
        ":value");
    static_assert(hpx::is_parallel_execution_policy<
                      std::execution::parallel_policy>::value,
        "hpx::is_parallel_execution_policy<std::execution::parallel_policy>::"
        "value");
    static_assert(hpx::is_parallel_execution_policy<
                      std::execution::parallel_unsequenced_policy>::value,
        "hpx::is_parallel_execution_policy<std::execution::parallel_"
        "unsequenced_policy>::value");

#if defined(HPX_HAVE_CXX20_STD_EXECUTION_POLICES)
    static_assert(
        hpx::is_execution_policy<std::execution::unsequenced_policy>::value,
        "hpx::is_execution_policy<std::execution::unsequenced_policy>::value");
    static_assert(hpx::is_sequenced_execution_policy<
                      std::execution::unsequenced_policy>::value,
        "hpx::is_sequenced_execution_policy<std::execution::unsequenced_policy>"
        "::value");
    static_assert(!hpx::is_parallel_execution_policy<
                      std::execution::unsequenced_policy>::value,
        "!hpx::is_parallel_execution_policy<std::execution::unsequenced_policy>"
        "::value");
#endif
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    static_checks();

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::local::init(hpx_main, argc, argv), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
