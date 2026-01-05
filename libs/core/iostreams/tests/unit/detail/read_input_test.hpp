//  Copyright (c) 2026 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// (C) Copyright 2008 CodeRage, LLC (turkanis at coderage dot com)
// (C) Copyright 2004-2007 Jonathan Turkanis

// See http://www.boost.org/libs/iostreams for documentation.

#pragma once

#include <hpx/modules/iostreams.hpp>
#include <hpx/modules/testing.hpp>

#include <fstream>

#include "detail/temp_file.hpp"
#include "detail/verification.hpp"

void read_input_test()
{
    using namespace std;
    using namespace hpx;
    using namespace hpx::iostreams;
    using namespace hpx::iostreams::test;

    test_file test1;
    test_file test2;

    {
        filtering_istream first(file_source(test1.name()), 0);
        ifstream second(test2.name().c_str());
        HPX_TEST_MSG(compare_streams_in_chars(first, second),
            "failed reading from filtering_istream in chars with no buffer");
    }

    {
        filtering_istream first(file_source(test1.name()), 0);
        ifstream second(test2.name().c_str());
        HPX_TEST_MSG(compare_streams_in_chunks(first, second),
            "failed reading from filtering_istream in chunks with no buffer");
    }

    {
        filtering_istream first(file_source(test1.name()));
        ifstream second(test2.name().c_str());
        HPX_TEST_MSG(compare_streams_in_chars(first, second),
            "failed reading from filtering_istream in chars with buffer");
    }

    {
        filtering_istream first(file_source(test1.name()));
        ifstream second(test2.name().c_str());
        HPX_TEST_MSG(compare_streams_in_chunks(first, second),
            "failed reading from filtering_istream in chunks with buffer");
    }
}
