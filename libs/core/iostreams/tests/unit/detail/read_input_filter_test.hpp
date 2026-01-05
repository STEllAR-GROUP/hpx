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

#include "detail/filters.hpp"
#include "detail/sequence.hpp"
#include "detail/temp_file.hpp"
#include "detail/verification.hpp"

void read_input_filter_test()
{
    using namespace std;
    using namespace hpx::iostreams;
    using namespace hpx::iostreams::test;

    test_file test;
    uppercase_file upper;

    {
        filtering_istream first;
        first.push(toupper_filter<input>());
        first.push(file_source(test.name()));
        ifstream second(upper.name().c_str());
        HPX_TEST_MSG(compare_streams_in_chars(first, second),
            "failed reading from a filtering_istream in chars with an "
            "input filter");
    }

    {
        filtering_istream first;
        first.push(toupper_filter<input>());
        first.push(file_source(test.name()));
        ifstream second(upper.name().c_str());
        HPX_TEST_MSG(compare_streams_in_chunks(first, second),
            "failed reading from a filtering_istream in chunks with an "
            "input filter");
    }

    {
        filtering_istream first;
        first.push(toupper_multichar_filter<input>(), 0);
        first.push(file_source(test.name()));
        ifstream second(upper.name().c_str());
        HPX_TEST_MSG(compare_streams_in_chars(first, second),
            "failed reading from a filtering_istream in chars with a "
            "multichar input filter with no buffer");
    }

    {
        filtering_istream first;
        first.push(toupper_multichar_filter<input>(), 0);
        first.push(file_source(test.name()));
        ifstream second(upper.name().c_str());
        HPX_TEST_MSG(compare_streams_in_chunks(first, second),
            "failed reading from a filtering_istream in chunks with a "
            "multichar input filter with no buffer");
    }

    {
        test_file src;
        filtering_istream first;
        first.push(toupper_multichar_filter<input>());
        first.push(file_source(src.name()));
        ifstream second(upper.name().c_str());
        HPX_TEST_MSG(compare_streams_in_chars(first, second),
            "failed reading from a filtering_istream in chars with a "
            "multichar input filter");
    }

    {
        test_file src;
        filtering_istream first;
        first.push(toupper_multichar_filter<input>());
        first.push(file_source(src.name()));
        ifstream second(upper.name().c_str());
        HPX_TEST_MSG(compare_streams_in_chunks(first, second),
            "failed reading from a filtering_istream in chunks with a "
            "multichar input filter");
    }
}
