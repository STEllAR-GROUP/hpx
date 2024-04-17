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
#ifdef HPX_HAVE_STDEXEC
            hpx::execution::experimental::forwarding_query_t,
#else
            hpx::execution::experimental::forwarding_scheduler_query_t,
#endif
            non_query_t) noexcept
        {
            return true;
        }
    } non_query{};

}    // namespace mylib

int main()
{
#ifdef HPX_HAVE_STDEXEC
    static_assert(hpx::execution::experimental::forwarding_query(
#else
    static_assert(hpx::execution::experimental::forwarding_scheduler_query(
#endif
                      mylib::non_query) == true,
        "non_query CPO is user implemented to return true");

#ifndef HPX_HAVE_STDEXEC
    static_assert(hpx::execution::experimental::forwarding_query(
                  hpx::execution::experimental::get_scheduler) == false,
              "invokes tag_fallback which returns false by default");
#endif

    return hpx::util::report_errors();
}
