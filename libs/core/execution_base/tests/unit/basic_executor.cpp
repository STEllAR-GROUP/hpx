//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/execution_base/sender.hpp>
#include <hpx/modules/testing.hpp>
#include <type_traits>

#include <cstddef>

static std::size_t member_execute_calls = 0;
static std::size_t tag_invoke_execute_calls = 0;

struct non_executor_1
{
};

struct non_executor_2
{
    template <typename F>
    void execute(F&&)
    {
        ++member_execute_calls;
    }
};

struct executor_1
{
    template <typename F>
    void execute(F&&)
    {
        ++member_execute_calls;
    }

    bool operator==(executor_1 const&) const noexcept
    {
        return true;
    }

    bool operator!=(executor_1 const&) const noexcept
    {
        return false;
    }
};

struct executor_2
{
    bool operator==(executor_2 const&) const noexcept
    {
        return true;
    }

    bool operator!=(executor_2 const&) const noexcept
    {
        return false;
    }
};

template <typename F>
void tag_invoke(hpx::execution::experimental::execute_t, executor_2, F&&)
{
    ++tag_invoke_execute_calls;
}

struct executor_3
{
    template <typename F>
    void execute(F&&)
    {
        ++member_execute_calls;
    }

    bool operator==(executor_3 const&) const noexcept
    {
        return true;
    }

    bool operator!=(executor_3 const&) const noexcept
    {
        return false;
    }
};

template <typename F>
void tag_invoke(hpx::execution::experimental::execute_t, executor_3, F&&)
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

void f_fun_1(){};

void f_fun_2(int){};

int main()
{
    using hpx::execution::experimental::is_executor;
    using hpx::execution::experimental::is_executor_of;

    static_assert(!is_executor<non_executor_1>::value,
        "non_executor_1 is not an executor");
    static_assert(!is_executor<non_executor_2>::value,
        "non_executor_2 is not an executor");
    static_assert(is_executor<executor_1>::value, "executor_1 is an executor");
    static_assert(is_executor<executor_2>::value, "executor_2 is an executor");
    static_assert(is_executor<executor_3>::value, "executor_3 is an executor");

    static_assert(!is_executor_of<non_executor_1, f_struct_1>::value,
        "non_executor_1 is not an executor of f_struct_1");
    static_assert(!is_executor_of<non_executor_2, f_struct_1>::value,
        "non_executor_2 is not an executor of f_struct_1");
    static_assert(is_executor_of<executor_1, f_struct_1>::value,
        "executor_1 is an executor of f_struct_1");
    static_assert(is_executor_of<executor_2, f_struct_1>::value,
        "executor_2 is an executor of f_struct_1");
    static_assert(is_executor_of<executor_3, f_struct_1>::value,
        "executor_3 is an executor of f_struct_1");

    static_assert(!is_executor_of<non_executor_1, f_struct_2>::value,
        "non_executor_1 is not an executor of f_struct");
    static_assert(!is_executor_of<non_executor_2, f_struct_2>::value,
        "non_executor_2 is not an executor of f_struct");
    static_assert(!is_executor_of<executor_1, f_struct_2>::value,
        "executor_1 is not an executor of f_struct_nonarchetypical");
    static_assert(!is_executor_of<executor_2, f_struct_2>::value,
        "executor_2 is not an executor of f_struct_nonarchetypical");
    static_assert(!is_executor_of<executor_3, f_struct_2>::value,
        "executor_3 is not an executor of f_struct_nonarchetypical");

    static_assert(!is_executor_of<non_executor_1, f_struct_3>::value,
        "non_executor_1 is not an executor of f_struct_2");
    static_assert(!is_executor_of<non_executor_2, f_struct_3>::value,
        "non_executor_2 is not an executor of f_struct_2");
    static_assert(is_executor_of<executor_1, f_struct_3>::value,
        "executor_1 is an executor of f_struct_2");
    static_assert(is_executor_of<executor_2, f_struct_3>::value,
        "executor_2 is an executor of f_struct_2");
    static_assert(is_executor_of<executor_3, f_struct_3>::value,
        "executor_3 is an executor of f_struct_2");

    static_assert(!is_executor_of<non_executor_1, decltype(f_fun_1)>::value,
        "non_executor_1 is not an executor of f_fun_1");
    static_assert(!is_executor_of<non_executor_2, decltype(f_fun_1)>::value,
        "non_executor_2 is not an executor of f_fun_1");
    static_assert(is_executor_of<executor_1, decltype(f_fun_1)>::value,
        "executor_1 is an executor of f_fun_1");
    static_assert(is_executor_of<executor_2, decltype(f_fun_1)>::value,
        "executor_2 is an executor of f_fun_1");
    static_assert(is_executor_of<executor_3, decltype(f_fun_1)>::value,
        "executor_3 is an executor of f_fun_1");

    static_assert(!is_executor_of<non_executor_1, decltype(f_fun_2)>::value,
        "non_executor_1 is not an executor of f_fun_2");
    static_assert(!is_executor_of<non_executor_2, decltype(f_fun_2)>::value,
        "non_executor_2 is not an executor of f_fun_2");
    static_assert(!is_executor_of<executor_1, decltype(f_fun_2)>::value,
        "executor_1 is an executor of f_fun_2");
    static_assert(!is_executor_of<executor_2, decltype(f_fun_2)>::value,
        "executor_2 is an executor of f_fun_2");
    static_assert(!is_executor_of<executor_3, decltype(f_fun_2)>::value,
        "executor_3 is an executor of f_fun_2");

    executor_1 e1;
    hpx::execution::experimental::execute(e1, f_struct_1{});
    hpx::execution::experimental::execute(e1, f_struct_3{});
    hpx::execution::experimental::execute(e1, &f_fun_1);
    HPX_TEST_EQ(member_execute_calls, std::size_t(3));
    HPX_TEST_EQ(tag_invoke_execute_calls, std::size_t(0));

    executor_2 e2;
    hpx::execution::experimental::execute(e2, f_struct_1{});
    hpx::execution::experimental::execute(e2, f_struct_3{});
    hpx::execution::experimental::execute(e2, &f_fun_1);
    HPX_TEST_EQ(member_execute_calls, std::size_t(3));
    HPX_TEST_EQ(tag_invoke_execute_calls, std::size_t(3));

    executor_3 e3;
    hpx::execution::experimental::execute(e3, f_struct_1{});
    hpx::execution::experimental::execute(e3, f_struct_3{});
    hpx::execution::experimental::execute(e3, &f_fun_1);
    HPX_TEST_EQ(member_execute_calls, std::size_t(6));
    HPX_TEST_EQ(tag_invoke_execute_calls, std::size_t(3));

    return hpx::util::report_errors();
}
