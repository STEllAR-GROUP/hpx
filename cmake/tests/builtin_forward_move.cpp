//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// test for std::move and std::forward being built in functions

#if defined(_MSC_VER)

#if !defined(__has_cpp_attribute)
#define __has_cpp_attribute(x) 0
#endif

#if __has_cpp_attribute(msvc::intrinsic)
#define MOVE_FORWARD_ARE_BUILTINS
#endif

#elif defined(__clang__) || defined(__gcc__)

#if !defined(__has_builtin)
#define __has_builtin(x) 0
#endif

#if __has_builtin(move) && __has_builtin(forward)
#define MOVE_FORWARD_ARE_BUILTINS
#endif

#endif

#if !defined(MOVE_FORWARD_ARE_BUILTINS)
#error "std::forward and/or std::move are not implemented as built in functions"
#endif

int main()
{
    return 0;
}
