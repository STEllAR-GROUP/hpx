//  Copyright (c) 2015-2017 Francisco Jose Tapia
//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/parallel/util/nbits.hpp>

int main(int, char*[])
{
    using namespace hpx::parallel::util;

    HPX_TEST(nbits32(63UL) == 6);
    HPX_TEST(nbits32(64UL) == 7);
    HPX_TEST(nbits32(65UL) == 7);

    HPX_TEST(nbits64(63ULL) == 6);
    HPX_TEST(nbits64(64ULL) == 7);
    HPX_TEST(nbits64(65ULL) == 7);

    return hpx::util::report_errors();
};
