//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Test if decltype on a lambda capture correctly detects rvalues. The rvalue
// overload should be used.

constexpr bool g(int&)
{
    return false;
}
constexpr bool g(int&&)
{
    return true;
}

template <typename T>
constexpr bool f(T&& t)
{
    return [&]() { return g(static_cast<decltype(t)&&>(t)); }();
}
int main()
{
    static_assert(f(3));
}
