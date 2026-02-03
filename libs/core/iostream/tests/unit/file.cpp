//  Copyright (c) 2026 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// (C) Copyright 2008 CodeRage, LLC (turkanis at coderage dot com)
// (C) Copyright 2004-2007 Jonathan Turkanis

// See http://www.boost.org/libs/iostreams for documentation.

#include <hpx/hpx_main.hpp>
#include <hpx/modules/iostream.hpp>
#include <hpx/modules/testing.hpp>

#include "detail/temp_file.hpp"
#include "detail/verification.hpp"

using namespace hpx::iostream;
using namespace hpx::iostream::test;

void file_test()
{
    test_file test;

    //--------------Test file_source------------------------------------------//
    {
        file_source f(test.name());
        HPX_TEST(f.is_open());
        f.close();
        HPX_TEST(!f.is_open());
        f.open(test.name());
        HPX_TEST(f.is_open());
    }

    //--------------Test file_sink--------------------------------------------//
    {
        file_sink f(test.name());
        HPX_TEST(f.is_open());
        f.close();
        HPX_TEST(!f.is_open());
        f.open(test.name());
        HPX_TEST(f.is_open());
    }

    //--------------Test file-------------------------------------------------//
    {
        file f(test.name());
        HPX_TEST(f.is_open());
        f.close();
        HPX_TEST(!f.is_open());
        f.open(test.name());
        HPX_TEST(f.is_open());
    }
}

int main(int, char*[])
{
    file_test();
    return hpx::util::report_errors();
}
