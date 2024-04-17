//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/execution/queries/get_scheduler.hpp>
#include <hpx/execution_base/get_env.hpp>
#include <hpx/functional/tag_invoke.hpp>
#include <hpx/modules/testing.hpp>

#include <exception>
#include <string>
#include <type_traits>
#include <utility>

namespace mylib {

#ifdef HPX_HAVE_STDEXEC
    struct derived_forwarding_query_t
      : hpx::execution::experimental::forwarding_query_t {
    } derived_forwarding_query{};
#endif

    inline constexpr struct non_query_t final
      : hpx::functional::tag<non_query_t>
    {
        friend constexpr auto tag_invoke(
#ifdef HPX_HAVE_STDEXEC
            hpx::execution::experimental::forwarding_query_t,
#else
            hpx::execution::experimental::forwarding_env_query_t,
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
                      mylib::non_query) == true,
#else
    static_assert(hpx::execution::experimental::forwarding_env_query(
                      mylib::non_query) == true,
#endif
        "non_query CPO is user implemented that returns true");
#ifdef HPX_HAVE_STDEXEC
    // P2300R8: "forwarding_query(execution::get_scheduler) is a core constant
    // expression and has value true."
    static_assert(hpx::execution::experimental::forwarding_query(
                      hpx::execution::experimental::get_scheduler) == true,
        "get scheduler is a forwarding query");
#else
    static_assert(hpx::execution::experimental::forwarding_env_query(
                      hpx::execution::experimental::get_scheduler) == false,
        "environmental query falls back to false");
#endif

#ifdef HPX_HAVE_STDEXEC
    static_assert(hpx::execution::experimental::forwarding_query(
                        mylib::derived_forwarding_query) == true,
        "derived_forwarding_query is a forwarding query");
#endif

    return hpx::util::report_errors();
}
