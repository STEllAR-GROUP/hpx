//  Copyright (c) 2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This example demonstrates the use of the facility invoke_function_action that
// allows to wrap arbitrary global functions into an action. Please note: this
// facility will work as expected only if the function address of the wrapped
// function is the same on all localities.

#include <hpx/hpx.hpp>
#include <hpx/hpx_main.hpp>

#include <iostream>

int call_me(int arg)
{
    return arg;
}

void void_call_me(int) {}

int main(int, char*[])
{
    // The function pointer is casted to a std::size_t to avoid compilation
    // problems complaining about raw pointers being used as action parameters.
    // The invoke_function facilities will cast this back to the correct
    // function pointer on the receiving end.
    {
        using action_type =
            hpx::components::server::invoke_function_action<decltype(&call_me),
                int>;

        hpx::future<int> result = hpx::async<action_type>(
            hpx::find_here(), reinterpret_cast<std::size_t>(&call_me), 42);

        std::cout << "the action invocation returned: " << result.get() << "\n";
    }

    {
        using action_type = hpx::components::server::invoke_function_action<
            decltype(&void_call_me), int>;

        hpx::future<void> result = hpx::async<action_type>(hpx::find_here(),
            reinterpret_cast<std::size_t>(&void_call_me), 42);

        result.get();
    }

    return 0;
}
