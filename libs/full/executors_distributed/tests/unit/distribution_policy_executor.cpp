//  Copyright (c) 2015-2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/util.hpp>
#include <hpx/modules/executors_distributed.hpp>
#include <hpx/modules/testing.hpp>

///////////////////////////////////////////////////////////////////////////////
hpx::id_type call()
{
    return hpx::find_here();
}

HPX_PLAIN_ACTION(call, call_action);

struct call_pfo
{
    hpx::id_type operator()() const
    {
        return hpx::find_here();
    }
};

///////////////////////////////////////////////////////////////////////////////
void test_distribution_policy_executor()
{
    using namespace hpx::parallel;

    for (hpx::id_type const& loc : hpx::find_all_localities())
    {
        auto exec =
            execution::make_distribution_policy_executor(hpx::colocated(loc));

        HPX_TEST_EQ(execution::async_execute(exec, call_pfo()).get(), loc);
        HPX_TEST_EQ(execution::async_execute(exec, call_action()).get(), loc);
    }

    for (hpx::id_type const& loc : hpx::find_all_localities())
    {
        auto exec =
            execution::make_distribution_policy_executor(hpx::colocated(loc));

        HPX_TEST_EQ(execution::sync_execute(exec, call_pfo()), loc);
        HPX_TEST_EQ(execution::sync_execute(exec, call_action()), loc);
    }
}

int main()
{
    test_distribution_policy_executor();
    return 0;
}
