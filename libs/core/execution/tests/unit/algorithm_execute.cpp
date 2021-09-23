//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/execution/algorithms/execute.hpp>
#include <hpx/modules/testing.hpp>

#include <cstddef>
#include <exception>
#include <type_traits>

static std::size_t friend_tag_dispatch_schedule_calls = 0;
static std::size_t tag_dispatch_execute_calls = 0;

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
        friend void tag_dispatch(hpx::execution::experimental::start_t,
            operation_state&) noexcept {};
    };

    template <typename R>
    friend operation_state tag_dispatch(
        hpx::execution::experimental::connect_t, sender&&, R&&) noexcept
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
    bool operator==(scheduler_1 const&) const noexcept
    {
        return true;
    }

    bool operator!=(scheduler_1 const&) const noexcept
    {
        return false;
    }
};

template <typename F>
void tag_dispatch(hpx::execution::experimental::execute_t, scheduler_2, F&&)
{
    ++tag_dispatch_execute_calls;
}

struct f_struct_1
{
    void operator()(){};
};

struct f_struct_2
{
    void operator()(int){};
};

struct f_struct_3
{
    void operator()(int = 42){};
};

void f_fun_1(){};

void f_fun_2(int){};

int main()
{
    {
        scheduler_1 s1;
        hpx::execution::experimental::execute(s1, f_struct_1{});
        hpx::execution::experimental::execute(s1, f_struct_3{});
        hpx::execution::experimental::execute(s1, &f_fun_1);
        HPX_TEST_EQ(friend_tag_dispatch_schedule_calls, std::size_t(3));
        HPX_TEST_EQ(tag_dispatch_execute_calls, std::size_t(0));
    }

    {
        scheduler_2 s2;
        hpx::execution::experimental::execute(s2, f_struct_1{});
        hpx::execution::experimental::execute(s2, f_struct_3{});
        hpx::execution::experimental::execute(s2, &f_fun_1);
        HPX_TEST_EQ(friend_tag_dispatch_schedule_calls, std::size_t(3));
        HPX_TEST_EQ(tag_dispatch_execute_calls, std::size_t(3));
    }

    return hpx::util::report_errors();
}
