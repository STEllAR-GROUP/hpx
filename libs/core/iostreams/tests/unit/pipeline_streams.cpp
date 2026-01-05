//  Copyright (c) 2026 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the hpx Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.hpx.org/LICENSE_1_0.txt)
//
// (C) Copyright 2008 CodeRage, LLC (turkanis at coderage dot com)
// (C) Copyright 2004-2007 Jonathan Turkanis

// See http://www.hpx.org/libs/iostreams for documentation.

#include <hpx/hpx_main.hpp>
#include <hpx/modules/iostreams.hpp>
#include <hpx/modules/testing.hpp>

#include <fstream>

#include "detail/filters.hpp"
#include "detail/sequence.hpp"
#include "detail/temp_file.hpp"
#include "detail/verification.hpp"

void pipeline_test()
{
    using namespace std;
    using namespace hpx;
    using namespace hpx::iostreams;
    using namespace hpx::iostreams::test;

    {
        test_file src;
        filtering_istream in1(
            toupper_filter<input>() | file_source(src.name()));
        filtering_istream in2(toupper_filter<input>() |
            toupper_filter<input>() | file_source(src.name()));
        filtering_istream in3(toupper_filter<input>() |
            toupper_filter<input>() | toupper_filter<input>() |
            file_source(src.name()));
        filtering_istream in4(toupper_filter<input>() |
            toupper_filter<input>() | toupper_filter<input>() |
            toupper_filter<input>() | file_source(src.name()));
        HPX_TEST(in1.size() == 2);
        HPX_TEST(in2.size() == 3);
        HPX_TEST(in3.size() == 4);
        HPX_TEST(in4.size() == 5);
    }

    {
        test_file src;
        uppercase_file upper;
        filtering_istream first(toupper_filter<input>() |
            toupper_multichar_filter<input>() |
            file_source(src.name(), in_mode));
        ifstream second(upper.name().c_str(), in_mode);
        HPX_TEST_MSG(compare_streams_in_chunks(first, second),
            "failed reading from a filtering_istream in chunks with a "
            "multichar input filter");
    }

    {
        temp_file dest;
        lowercase_file lower;
        filtering_ostream out(tolower_filter<output>() |
            tolower_multichar_filter<output>() |
            file_sink(dest.name(), out_mode));
        write_data_in_chunks(out);
        out.reset();
        HPX_TEST_MSG(compare_files(dest.name(), lower.name()),
            "failed writing to a filtering_ostream in chunks with a "
            "multichar output filter with no buffer");
    }
}

int main(int, char*[])
{
    pipeline_test();
    return hpx::util::report_errors();
}
