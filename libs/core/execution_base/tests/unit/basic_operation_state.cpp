//  Copyright (c) 2020 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/execution_base/operation_state.hpp>
#include <hpx/modules/testing.hpp>

#include <exception>
#include <string>
#include <type_traits>
#include <utility>

bool start_called = false;

namespace mylib {
    struct state_1
    {
    };

    struct state_2
    {
        void start();
    };

    struct state_3
    {
        void start() noexcept
        {
            start_called = true;
        }
    };

    struct state_4
    {
    };

    void tag_invoke(hpx::execution_base::start_t, state_4) {}

    struct state_5
    {
    };

    void tag_invoke(hpx::execution_base::start_t, state_5) noexcept
    {
        start_called = true;
    }
}    // namespace mylib

int main()
{
    static_assert(
        !hpx::execution_base::traits::is_operation_state<mylib::state_1>::value,
        "mylib::state_1 is not an operation state");
    static_assert(
        !hpx::execution_base::traits::is_operation_state<mylib::state_2>::value,
        "mylib::state_2 is not an operation state");
    static_assert(
        hpx::execution_base::traits::is_operation_state<mylib::state_3>::value,
        "mylib::state_3 is an operation state");
    static_assert(
        !hpx::execution_base::traits::is_operation_state<mylib::state_4>::value,
        "mylib::state_4 is not an operation state");
    static_assert(
        hpx::execution_base::traits::is_operation_state<mylib::state_5>::value,
        "mylib::state_5 is an operation state");

    {
        mylib::state_3 state;

        hpx::execution_base::start(state);
        HPX_TEST(start_called);
        start_called = false;
    }
    {
        mylib::state_5 state;

        hpx::execution_base::start(state);
        HPX_TEST(start_called);
        start_called = false;
    }

    return hpx::util::report_errors();
}
