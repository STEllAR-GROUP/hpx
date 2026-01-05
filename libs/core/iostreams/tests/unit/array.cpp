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
#include <hpx/modules/iostreams.hpp>
#include <hpx/modules/testing.hpp>

#include "detail/sequence.hpp"
#include "detail/temp_file.hpp"
#include "detail/verification.hpp"

void array_test()
{
    using namespace std;
    using namespace hpx::iostreams;
    using namespace hpx::iostreams::test;

    test_file test;

    //--------------stream<array_source>-------------------------------//
    {
        test_sequence<> seq;
        stream<array_source<char>> first(&seq[0], &seq[0] + seq.size());
        ifstream second(
            test.name().c_str(), std::ios_base::in | std::ios_base::binary);
        HPX_TEST_MSG(compare_streams_in_chars(first, second),
            "failed reading from stream<array_source> in chars");
    }

    {
        test_sequence<> seq;
        stream<array_source<char>> first(&seq[0], &seq[0] + seq.size());
        ifstream second(
            test.name().c_str(), std::ios_base::in | std::ios_base::binary);
        HPX_TEST_MSG(compare_streams_in_chunks(first, second),
            "failed reading from stream<array_source> in chunks");
    }

    //--------------stream<array_sink>---------------------------------//
    {
        vector<char> first(data_reps * data_length(), '?');
        stream<array_sink<char>> out(&first[0], &first[0] + first.size());
        write_data_in_chars(out);
        ifstream second(
            test.name().c_str(), std::ios_base::in | std::ios_base::binary);
        HPX_TEST_MSG(compare_container_and_stream(first, second),
            "failed writing to stream<array_sink> in chars");
    }

    {
        vector<char> first(data_reps * data_length(), '?');
        stream<array_sink<char>> out(&first[0], &first[0] + first.size());
        write_data_in_chunks(out);
        ifstream second(
            test.name().c_str(), std::ios_base::in | std::ios_base::binary);
        HPX_TEST_MSG(compare_container_and_stream(first, second),
            "failed writing to stream<array_sink> in chunks");
    }

    //--------------random access---------------------------------------------//
    {
        vector<char> first(data_reps * data_length(), '?');
        stream<hpx::iostreams::array<char>> io(&first[0], &first[0] + first.size());
        HPX_TEST_MSG(test_seekable_in_chars(io),
            "failed seeking within stream<array>, in chars");
    }

    {
        vector<char> first(data_reps * data_length(), '?');
        stream<hpx::iostreams::array<char>> io(&first[0], &first[0] + first.size());
        HPX_TEST_MSG(test_seekable_in_chars(io),
            "failed seeking within stream<array>, in chunks");
    }
}

int main(int, char*[])
{
    array_test();
    return hpx::util::report_errors();
}
