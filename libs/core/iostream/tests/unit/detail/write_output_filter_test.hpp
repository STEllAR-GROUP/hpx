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

void write_output_filter_test()
{
    using namespace std;
    using namespace hpx::iostream;
    using namespace hpx::iostream::test;

    lowercase_file lower;

    {
        temp_file dest;
        filtering_ostream out;
        out.push(tolower_filter<output>());
        out.push(file_sink(dest.name(), out_mode));
        write_data_in_chars(out);
        out.reset();
        HPX_TEST_MSG(compare_files(dest.name(), lower.name()),
            "failed writing to a filtering_ostream in chars with an "
            "output filter");
    }

    {
        temp_file dest;
        filtering_ostream out;
        out.push(tolower_filter<output>());
        out.push(file_sink(dest.name(), out_mode));
        write_data_in_chunks(out);
        out.reset();
        HPX_TEST_MSG(compare_files(dest.name(), lower.name()),
            "failed writing to a filtering_ostream in chunks with an "
            "output filter");
    }

    {
        temp_file dest;
        filtering_ostream out;
        out.push(tolower_multichar_filter<output>(), 0);
        out.push(file_sink(dest.name(), out_mode));
        write_data_in_chars(out);
        out.reset();
        HPX_TEST_MSG(compare_files(dest.name(), lower.name()),
            "failed writing to a filtering_ostream in chars with a "
            "multichar output filter with no buffer");
    }

    {
        temp_file dest;
        filtering_ostream out;
        out.push(tolower_multichar_filter<output>(), 0);
        out.push(file_sink(dest.name(), out_mode));
        write_data_in_chunks(out);
        out.reset();
        HPX_TEST_MSG(compare_files(dest.name(), lower.name()),
            "failed writing to a filtering_ostream in chunks with a "
            "multichar output filter with no buffer");
    }

    {
        temp_file dest;
        filtering_ostream out;
        out.push(tolower_multichar_filter<output>());
        out.push(file_sink(dest.name(), out_mode));
        write_data_in_chars(out);
        out.reset();
        HPX_TEST_MSG(compare_files(dest.name(), lower.name()),
            "failed writing to a filtering_ostream in chars with a "
            "multichar output filter");
    }

    {
        temp_file dest;
        filtering_ostream out;
        out.push(tolower_multichar_filter<output>());
        out.push(file_sink(dest.name(), out_mode));
        write_data_in_chunks(out);
        out.reset();
        HPX_TEST_MSG(compare_files(dest.name(), lower.name()),
            "failed writing to a filtering_ostream in chunks with a "
            "multichar output filter");
    }
}
