//  Copyright (c) 2022 Shreyas Atre
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/modules/execution_base.hpp>
#include <hpx/modules/testing.hpp>

struct scheduler
{
    friend constexpr bool tag_invoke(
        hpx::execution_base::this_thread::execute_may_block_caller_t,
        const scheduler&) noexcept
    {
        return false;
    }
};

int main()
{
    {
        scheduler s1;
        static_assert(
            hpx::execution_base::this_thread::execute_may_block_caller(s1) ==
                false,
            "CPO returns false");
    }

    return hpx::util::report_errors();
}
