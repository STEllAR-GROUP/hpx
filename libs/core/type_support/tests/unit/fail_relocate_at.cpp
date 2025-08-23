//  Copyright (c) 2023 Isidoros Tsaousis-Seiras
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test should fail to compile

#include <hpx/init.hpp>
#include <hpx/type_support/relocate_at.hpp>

using hpx::experimental::relocate_at;

int main(int argc, char* argv[])
{
    int a[10];
    int b[10];

    int (*p)[10] = &a;
    int (*q)[10] = &b;

    relocate_at(p, q);
}
