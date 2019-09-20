//  Copyright (c) 2016 Agustin Berge
//  Copyright (c) 2019 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// this tests for LWG3018 being resolved

#include <memory>

typedef void (function_type)();

void test() {}

struct deleter
{
     void operator()(function_type*) {}
};

int main()
{
    auto sp = std::shared_ptr<function_type>(&test, deleter{});
    (void)sp;
}
