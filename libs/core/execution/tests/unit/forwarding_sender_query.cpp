//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/modules/execution.hpp>
#include <hpx/modules/execution_base.hpp>
#include <hpx/modules/tag_invoke.hpp>
#include <hpx/modules/testing.hpp>

#include <exception>
#include <string>
#include <type_traits>
#include <utility>

namespace ex = hpx::execution::experimental;

namespace mylib {

    inline constexpr struct query_t final : hpx::functional::tag<query_t>
    {
        friend constexpr auto tag_invoke(
#if defined(HPX_HAVE_STDEXEC)
            ex::forwarding_query_t, query_t) noexcept
#else
            ex::forwarding_sender_query_t, query_t) noexcept
#endif
        {
            return true;
        }
    } query{};

    inline constexpr struct non_query_t
    {
    } non_query{};

}    // namespace mylib

int main()
{
#if defined(HPX_HAVE_STDEXEC)
    static_assert(ex::forwarding_query(mylib::query) == true,
#else
    static_assert(ex::forwarding_sender_query(mylib::query) == true,
#endif
        "non_query CPO is user implemented to return true");

#if defined(HPX_HAVE_STDEXEC)
    static_assert(ex::forwarding_query(mylib::non_query) == false,
#else
    static_assert(ex::forwarding_sender_query(mylib::non_query) == false,
#endif
        "invokes tag_fallback which returns false by default");

#if defined(HPX_HAVE_STDEXEC)
    static_assert(ex::forwarding_query(
#else
    static_assert(ex::forwarding_sender_query(
#endif
                      ex::get_completion_scheduler<ex::set_value_t>) == true,
        "invokes CPO specialization that returns true");

#if defined(HPX_HAVE_STDEXEC)
    static_assert(ex::forwarding_query(
#else
    static_assert(ex::forwarding_sender_query(
#endif
                      ex::get_completion_scheduler<ex::set_error_t>) == true,
        "invokes CPO specialization that returns true");

#if defined(HPX_HAVE_STDEXEC)
    static_assert(ex::forwarding_query(
#else
    static_assert(ex::forwarding_sender_query(
#endif
                      ex::get_completion_scheduler<ex::set_stopped_t>) == true,
        "invokes CPO specialization that returns true");

    return hpx::util::report_errors();
}
