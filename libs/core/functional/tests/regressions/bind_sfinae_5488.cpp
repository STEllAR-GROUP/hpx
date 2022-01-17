//  Copyright (c) 2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// #5488: hpx::util::bind doesn't bounds-check placeholders

#include <hpx/modules/functional.hpp>
#include <hpx/modules/testing.hpp>

#include <type_traits>

template <typename F, std::enable_if_t<hpx::is_invocable_v<F>, int> = 0>
auto test(F&& f)
{
    return f();
}

template <typename F, std::enable_if_t<hpx::is_invocable_v<F, int>, int> = 0>
auto test(F&& f)
{
    return f(42);
}

void foo(int) {}

int main()
{
    test(hpx::util::bind(foo, hpx::util::placeholders::_1));
    return hpx::util::report_errors();
}
