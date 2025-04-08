//  Copyright (c) 2025 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/datapar.hpp>
#include <hpx/execution.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/program_options.hpp>

#include <type_traits>

template <typename Derived>
struct custom_base
{
};

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy>
void test_base(ExPolicy&& policy)
{
    using policy_t = std::decay_t<ExPolicy>;

    static_assert(std::is_base_of_v<custom_base<policy_t>, policy_t>,
        "must have custom_base<policy_t> as base class");

    (void) policy;
}

void test_base()
{
    using namespace hpx::execution;

    test_base(basic_sequenced_policy<custom_base>{});
    test_base(basic_parallel_policy<custom_base>{});
    test_base(basic_unsequenced_policy<custom_base>{});
    test_base(basic_parallel_unsequenced_policy<custom_base>{});
    test_base(basic_sequenced_task_policy<custom_base>{});
    test_base(basic_parallel_task_policy<custom_base>{});
    test_base(basic_unsequenced_task_policy<custom_base>{});
    test_base(basic_parallel_unsequenced_task_policy<custom_base>{});
#if defined(HPX_HAVE_DATAPAR)
    test_base(basic_simd_task_policy<custom_base>{});
    test_base(basic_par_simd_task_policy<custom_base>{});
    test_base(basic_simd_task_policy<custom_base>{});
    test_base(basic_par_simd_task_policy<custom_base>{});
#endif
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(hpx::program_options::variables_map&)
{
    test_base();

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::local::init(hpx_main, argc, argv), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
