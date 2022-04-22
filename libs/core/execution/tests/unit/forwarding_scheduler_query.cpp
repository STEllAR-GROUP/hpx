//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/execution/queries/get_scheduler.hpp>
#include <hpx/functional/tag_invoke.hpp>
#include <hpx/modules/testing.hpp>

#include <exception>
#include <string>
#include <type_traits>
#include <utility>

namespace mylib {

    inline constexpr struct non_query_t final
      : hpx::functional::tag<non_query_t>
    {
        friend constexpr auto tag_invoke(
            hpx::execution::experimental::forwarding_scheduler_query_t,
            non_query_t) noexcept
        {
            return false;
        }
    } non_query{};

}    // namespace mylib

int main()
{
    static_assert(!hpx::execution::experimental::forwarding_scheduler_query(
                      mylib::non_query),
        "non_query is not a scheduler query");

    static_assert(hpx::execution::experimental::forwarding_scheduler_query(
                      hpx::execution::experimental::get_scheduler),
        "get_scheduler is a scheduler query");

    return hpx::util::report_errors();
}
