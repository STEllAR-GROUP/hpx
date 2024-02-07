//  Copyright (c) 2024 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// test for availability of std::experimental::scope_xxx

#include <experimental/scope>

#if !defined(__cpp_lib_experimental_scope)
#  error "__cpp_lib_experimental_scope not defined, assume scope_exit etc. is not supported"
#endif

int main()
{
    std::experimental::scope_exit se([] {});
    std::experimental::scope_failure sf([] {});
    std::experimental::scope_success ss([] {});

    return 0;
}
