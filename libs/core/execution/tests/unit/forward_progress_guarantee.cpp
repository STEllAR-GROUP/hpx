//  Copyright (c) 2022 Shreyas Atre
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/execution/queries/get_scheduler.hpp>
#include <hpx/execution_base/completion_scheduler.hpp>
#include <hpx/modules/testing.hpp>

#include <exception>
#include <utility>

namespace mylib {

    // CPO
    struct inline_scheduler_0
    {
        constexpr friend HPX_FORCEINLINE auto tag_invoke(
            hpx::execution::experimental::get_forward_progress_guarantee_t,
            const inline_scheduler_0&) noexcept
        {
            return hpx::execution::experimental::forward_progress_guarantee::
                concurrent;
        }

    } scheduler{};

    // CPO
    struct inline_scheduler_1
    {
        /// With the same user-defined tag_invoke overload, the user-defined
        /// overload will now be used if it is a match even if it isn't an exact
        /// match.
        /// This is because tag_fallback will dispatch to tag_fallback_invoke only
        /// if there are no matching tag_invoke overloads.
        constexpr friend auto tag_invoke(
            hpx::execution::experimental::get_forward_progress_guarantee_t,
            inline_scheduler_1) noexcept
        {
            return true;
        }
    } scheduler_custom{};

}    // namespace mylib

int main()
{
    using namespace mylib;
    static_assert(hpx::execution::experimental::get_forward_progress_guarantee(
                      scheduler) ==
            hpx::execution::experimental::forward_progress_guarantee::
                concurrent,
        "forward_progress_guarantee should return concurrent");

    static_assert(hpx::execution::experimental::get_forward_progress_guarantee(
                      scheduler_custom),
        "CPO should invoke user tag_invoke");

    return hpx::util::report_errors();
}
