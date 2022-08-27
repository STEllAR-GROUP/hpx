//  Copyright (c) 2020 ETH Zurich
//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/execution/algorithms/execute.hpp>
#include <hpx/modules/testing.hpp>

#include <cstddef>
#include <exception>
#include <type_traits>

namespace ex = hpx::execution::experimental;

static std::size_t friend_tag_invoke_schedule_calls = 0;
static std::size_t tag_invoke_execute_calls = 0;

struct sender
{
    template <typename Env>
    friend auto tag_invoke(ex::get_completion_signatures_t, sender const&, Env)
        -> ex::completion_signatures<ex::set_value_t(),
            ex::set_error_t(std::exception_ptr)>;

    struct operation_state
    {
        friend void tag_invoke(ex::start_t, operation_state&) noexcept {};
    };

    template <typename R>
    friend operation_state tag_invoke(ex::connect_t, sender&&, R&&) noexcept
    {
        return {};
    }
};

struct scheduler_1
{
    friend sender tag_invoke(ex::schedule_t, scheduler_1)
    {
        ++friend_tag_invoke_schedule_calls;
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
void tag_invoke(ex::execute_t, scheduler_2, F&&)
{
    ++tag_invoke_execute_calls;
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

void f_fun_1() {}

void f_fun_2(int) {}

int main()
{
    {
        scheduler_1 s1;
        ex::execute(s1, f_struct_1{});
        ex::execute(s1, f_struct_3{});
        ex::execute(s1, &f_fun_1);
        HPX_TEST_EQ(friend_tag_invoke_schedule_calls, std::size_t(3));
        HPX_TEST_EQ(tag_invoke_execute_calls, std::size_t(0));
    }

    {
        scheduler_2 s2;
        ex::execute(s2, f_struct_1{});
        ex::execute(s2, f_struct_3{});
        ex::execute(s2, &f_fun_1);
        HPX_TEST_EQ(friend_tag_invoke_schedule_calls, std::size_t(3));
        HPX_TEST_EQ(tag_invoke_execute_calls, std::size_t(3));
    }

    return hpx::util::report_errors();
}
