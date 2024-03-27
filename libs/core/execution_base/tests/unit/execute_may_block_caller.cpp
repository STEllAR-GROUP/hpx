//  Copyright (c) 2022 Shreyas Atre
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/modules/execution_base.hpp>
#include <hpx/modules/testing.hpp>

namespace ex = hpx::execution::experimental;
namespace tt = hpx::this_thread::experimental;

struct scheduler
{
    friend constexpr void tag_invoke(ex::schedule_t, scheduler) noexcept {}

    friend constexpr bool tag_invoke(
        tt::execute_may_block_caller_t, scheduler) noexcept
    {
        return false;
    }

    friend constexpr bool operator==(scheduler, scheduler) noexcept
    {
        return true;
    }

    friend constexpr bool operator!=(scheduler, scheduler) noexcept
    {
        return false;
    }
};

int main()
{
#ifndef HPX_HAVE_STDEXEC
    /*TODO: This is missing a lot to pass the scheduler concept check*/
    static_assert(ex::is_scheduler_v<scheduler>);
#endif

    {
        constexpr scheduler s1{};
        static_assert(
            !tt::execute_may_block_caller(s1), "CPO should return false");
    }

    return hpx::util::report_errors();
}
