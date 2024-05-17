//  Copyright (c) 2022 Shreyas Atre
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/modules/execution_base.hpp>
#include <hpx/modules/testing.hpp>

#include "algorithm_test_utils.hpp"

namespace ex = hpx::execution::experimental;
namespace tt = hpx::this_thread::experimental;

struct this_test_example_scheduler
{
#ifdef HPX_HAVE_STDEXEC
    struct example_sender
    {
        using is_sender = void;
        using completion_signatures = ex::completion_signatures<>;

        friend env_with_scheduler<this_test_example_scheduler> tag_invoke(
            ex::get_env_t, example_sender const&) noexcept
        {
            return {};
        }
    };
#endif

#ifdef HPX_HAVE_STDEXEC
    friend constexpr example_sender tag_invoke(
        ex::schedule_t, this_test_example_scheduler) noexcept
    {
        return {};
    }
#else
    friend constexpr voiud tag_invoke(
        ex::schedule_t, this_test_example_scheduler) noexcept
    {
    }
#endif

    friend constexpr bool tag_invoke(
        tt::execute_may_block_caller_t, this_test_example_scheduler) noexcept
    {
        return false;
    }

    friend constexpr bool operator==(
        this_test_example_scheduler, this_test_example_scheduler) noexcept
    {
        return true;
    }

    friend constexpr bool operator!=(
        this_test_example_scheduler, this_test_example_scheduler) noexcept
    {
        return false;
    }
};

int main()
{
    static_assert(ex::is_scheduler_v<this_test_example_scheduler>);

    {
        constexpr this_test_example_scheduler s1{};
        static_assert(
            !tt::execute_may_block_caller(s1), "CPO should return false");
    }

    return hpx::util::report_errors();
}
