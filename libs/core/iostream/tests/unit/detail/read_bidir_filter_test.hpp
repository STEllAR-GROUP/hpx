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

#include <hpx/modules/iostream.hpp>
#include <hpx/modules/testing.hpp>

#include <fstream>

#include "filters.hpp"
#include "sequence.hpp"
#include "temp_file.hpp"
#include "verification.hpp"

void read_bidirectional_filter_test()
{
    using namespace std;
    using namespace hpx::iostream;
    using namespace hpx::iostream::test;

    uppercase_file upper;

    {
        test_file src;
        temp_file dest;    // Dummy.
        filtering_stream<bidirectional> first;
        first.push(combine(toupper_filter<input>(), tolower_filter<output>()));
        first.push(combine(file_source(src.name()), file_sink(dest.name())));
        ifstream second(upper.name().c_str());
        HPX_TEST_MSG(compare_streams_in_chars(first, second),
            "failed reading from filtering_stream<bidirectional> in chars with "
            "an input filter");
    }

    {
        test_file src;
        temp_file dest;    // Dummy.
        filtering_stream<bidirectional> first;
        first.push(combine(toupper_filter<input>(), tolower_filter<output>()));
        first.push(combine(file_source(src.name()), file_sink(dest.name())));
        ifstream second(upper.name().c_str());
        HPX_TEST_MSG(compare_streams_in_chunks(first, second),
            "failed reading from filtering_stream<bidirectional> in chunks "
            "with an input filter");
    }

    //{
    //    test_file                        src;
    //    temp_file                        dest; // Dummy.
    //    filtering_stream<bidirectional>  first(
    //        combine(toupper_multichar_filter(), tolower_filter()), 0
    //    );
    //    first.push(
    //        combine(file_source(src.name()), file_sink(dest.name()))
    //    );
    //    ifstream second(upper.name().c_str());
    //    HPX_TEST_MSG(
    //        compare_streams_in_chars(first, second),
    //        "failed reading from filtering_stream<bidirectional> in chars with "
    //        "a multichar input filter with no buffer"
    //    );
    //}

    //{
    //    test_file                        src;
    //    temp_file                        dest; // Dummy.
    //    filtering_stream<bidirectional>  first(
    //        combine(toupper_multichar_filter(), tolower_filter()), 0
    //    );
    //    first.push(
    //        combine(file_source(src.name()), file_sink(dest.name()))
    //    );
    //    ifstream second(upper.name().c_str());
    //    HPX_TEST_MSG(
    //        compare_streams_in_chunks(first, second),
    //        "failed reading from filtering_stream<bidirectional> in chunks "
    //        "with a multichar input filter with no buffer"
    //    );
    //}

    //{
    //    test_file                        src;
    //    temp_file                        dest; // Dummy.
    //    filtering_stream<bidirectional>  first(
    //        combine(toupper_multichar_filter(), tolower_filter())
    //    );
    //    first.push(
    //        combine(file_source(src.name()), file_sink(dest.name()))
    //    );
    //    ifstream second(upper.name().c_str());
    //    HPX_TEST_MSG(
    //        compare_streams_in_chars(first, second),
    //        "failed reading from filtering_stream<bidirectional> in chars with a "
    //        "multichar input filter"
    //    );
    //}

    //{
    //    test_file                        src;
    //    temp_file                        dest; // Dummy.
    //    filtering_stream<bidirectional>  first(
    //        combine(toupper_multichar_filter(), tolower_filter())
    //    );
    //    first.push(
    //        combine(file_source(src.name()), file_sink(dest.name()))
    //    );
    //    ifstream second(upper.name().c_str());
    //    HPX_TEST_MSG(
    //        compare_streams_in_chunks(first, second),
    //        "failed reading from filtering_stream<bidirectional> in chunks "
    //        "with a multichar input filter"
    //    );
    //}
}
