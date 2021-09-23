//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/execution_base/sender.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/type_support/unused.hpp>

#include <cstddef>
#include <exception>
#include <type_traits>

static std::size_t friend_tag_dispatch_schedule_calls = 0;
static std::size_t tag_dispatch_schedule_calls = 0;

struct sender
{
    template <template <class...> class Tuple,
        template <class...> class Variant>
    using value_types = Variant<Tuple<>>;

    template <template <class...> class Variant>
    using error_types = Variant<std::exception_ptr>;

    static constexpr bool sends_done = false;

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

struct non_scheduler_1
{
};

struct non_scheduler_2
{
    void schedule() {}
};

struct non_scheduler_3
{
    friend sender tag_dispatch(
        hpx::execution::experimental::schedule_t, non_scheduler_3)
    {
        return {};
    }
};

struct scheduler_1
{
    friend sender tag_dispatch(
        hpx::execution::experimental::schedule_t, scheduler_1)
    {
        ++friend_tag_dispatch_schedule_calls;
        return {};
    }

    bool operator==(scheduler_1 const&) const noexcept
    {
        return true;
    }

    bool operator!=(scheduler_1 const&) const noexcept
    {
        return false;
    }
};

struct scheduler_2
{
    bool operator==(scheduler_2 const&) const noexcept
    {
        return true;
    }

    bool operator!=(scheduler_2 const&) const noexcept
    {
        return false;
    }
};

sender tag_dispatch(hpx::execution::experimental::schedule_t, scheduler_2)
{
    ++tag_dispatch_schedule_calls;
    return {};
}

int main()
{
    using hpx::execution::experimental::is_scheduler;

    static_assert(!is_scheduler<non_scheduler_1>::value,
        "non_scheduler_1 is not a scheduler");
    static_assert(!is_scheduler<non_scheduler_2>::value,
        "non_scheduler_2 is not a scheduler");
    static_assert(!is_scheduler<non_scheduler_3>::value,
        "non_scheduler_3 is not a scheduler");
    static_assert(
        is_scheduler<scheduler_1>::value, "scheduler_1 is a scheduler");
    static_assert(
        is_scheduler<scheduler_2>::value, "scheduler_2 is a scheduler");

    scheduler_1 s1;
    sender snd1 = hpx::execution::experimental::schedule(s1);
    HPX_UNUSED(snd1);
    HPX_TEST_EQ(friend_tag_dispatch_schedule_calls, std::size_t(1));
    HPX_TEST_EQ(tag_dispatch_schedule_calls, std::size_t(0));

    scheduler_2 s2;
    sender snd2 = hpx::execution::experimental::schedule(s2);
    HPX_UNUSED(snd2);
    HPX_TEST_EQ(friend_tag_dispatch_schedule_calls, std::size_t(1));
    HPX_TEST_EQ(tag_dispatch_schedule_calls, std::size_t(1));

    return hpx::util::report_errors();
}
