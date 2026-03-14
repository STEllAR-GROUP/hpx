//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/modules/execution_base.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/modules/type_support.hpp>

#include <cstddef>
#include <exception>
#include <type_traits>
#include "algorithm_test_utils.hpp"

namespace ex = hpx::execution::experimental;

static std::size_t friend_tag_invoke_schedule_calls = 0;
static std::size_t tag_invoke_schedule_calls = 0;

#if defined(HPX_HAVE_STDEXEC)
struct dummy_scheduler
{
};

template <typename Scheduler = dummy_scheduler>
#endif
struct example_sender_template
{
#if defined(HPX_HAVE_STDEXEC)
    using is_sender = void;

    using completion_signatures = ex::completion_signatures<ex::set_value_t(),
        ex::set_error_t(std::exception_ptr)>;

    friend env_with_scheduler<Scheduler> tag_invoke(
        ex::get_env_t, example_sender_template const&) noexcept
    {
        return {};
    }
#else
    template <template <class...> class Tuple,
        template <class...> class Variant>
    using value_types = Variant<Tuple<>>;

    template <template <class...> class Variant>
    using error_types = Variant<std::exception_ptr>;

    static constexpr bool sends_stopped = false;
#endif

    struct operation_state
    {
        void start() & noexcept {};
    };

    template <typename R>
    operation_state connect(R&&) && noexcept
    {
        return {};
    }
};
#if defined(HPX_HAVE_STDEXEC)
using example_sender = example_sender_template<>;
#else
using example_sender = example_sender_template;
#endif

struct non_scheduler_1
{
};

struct non_scheduler_2
{
    void schedule() {}
};

struct non_scheduler_3
{
    friend example_sender tag_invoke(ex::schedule_t, non_scheduler_3)
    {
        return {};
    }
};

struct scheduler_1
{
#if defined(HPX_HAVE_STDEXEC)
    using sender_1 = example_sender_template<scheduler_1>;

    friend sender_1 tag_invoke(ex::schedule_t, scheduler_1)
    {
        ++friend_tag_invoke_schedule_calls;
        return {};
    }
#else
    friend example_sender tag_invoke(ex::schedule_t, scheduler_1)
    {
        ++friend_tag_invoke_schedule_calls;
        return {};
    }
#endif

    constexpr bool operator==(scheduler_1 const&) const noexcept
    {
        return true;
    }

    constexpr bool operator!=(scheduler_1 const&) const noexcept
    {
        return false;
    }
};

struct scheduler_2
{
#if defined(HPX_HAVE_STDEXEC)
    using sender_2 = example_sender_template<scheduler_2>;
#endif

    constexpr bool operator==(scheduler_2 const&) const noexcept
    {
        return true;
    }

    constexpr bool operator!=(scheduler_2 const&) const noexcept
    {
        return false;
    }
};

#if defined(HPX_HAVE_STDEXEC)
example_sender_template<scheduler_2> tag_invoke(ex::schedule_t, scheduler_2)
#else
example_sender tag_invoke(ex::schedule_t, scheduler_2)
#endif
{
    ++tag_invoke_schedule_calls;
    return {};
}

int main()
{
    using ex::is_scheduler_v;

    static_assert(
        !is_scheduler_v<non_scheduler_1>, "non_scheduler_1 is not a scheduler");
#if !defined(HPX_HAVE_STDEXEC)
    // In P2300 the result of a `schedule` member function is mandated to be a sender,
    // so the evaluation of the following triggers a compilation error.
    static_assert(
        !is_scheduler_v<non_scheduler_2>, "non_scheduler_2 is not a scheduler");
#endif
    static_assert(
        !is_scheduler_v<non_scheduler_3>, "non_scheduler_3 is not a scheduler");
    static_assert(is_scheduler_v<scheduler_1>, "scheduler_1 is a scheduler");
    static_assert(is_scheduler_v<scheduler_2>, "scheduler_2 is a scheduler");

    scheduler_1 s1;
    auto snd1 = ex::schedule(s1);
    HPX_UNUSED(snd1);
    HPX_TEST_EQ(friend_tag_invoke_schedule_calls, std::size_t(1));
    HPX_TEST_EQ(tag_invoke_schedule_calls, std::size_t(0));

    scheduler_2 s2;
    auto snd2 = ex::schedule(s2);
    HPX_UNUSED(snd2);
    HPX_TEST_EQ(friend_tag_invoke_schedule_calls, std::size_t(1));
    HPX_TEST_EQ(tag_invoke_schedule_calls, std::size_t(1));

    static_assert(std::is_same_v<ex::schedule_result_t<scheduler_1>,
#if defined(HPX_HAVE_STDEXEC)
                      example_sender_template<scheduler_1>
#else
                      example_sender
#endif
                      >,
        "Result of scheduler is a example_sender");
    static_assert(std::is_same_v<ex::schedule_result_t<scheduler_2>,
#if defined(HPX_HAVE_STDEXEC)
                      example_sender_template<scheduler_2>
#else
                      example_sender
#endif
                      >,
        "Result of scheduler is a example_sender");

    return hpx::util::report_errors();
}
