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

#include <algorithm>
#include <fstream>

#include "detail/sequence.hpp"
#include "detail/temp_file.hpp"
#include "detail/verification.hpp"

using namespace std;
using namespace hpx::iostreams;
using namespace hpx::iostreams::test;

void direct_adapter_test()
{
    typedef hpx::iostreams::detail::direct_adapter<array_source<char>>
        indirect_array_source;
    typedef hpx::iostreams::detail::direct_adapter<array_sink<char>>
        indirect_array_sink;
    typedef hpx::iostreams::detail::direct_adapter<hpx::iostreams::array<char>>
        indirect_array;
    typedef stream<indirect_array_source> indirect_array_istream;
    typedef stream<indirect_array_sink> indirect_array_ostream;
    typedef stream<indirect_array> indirect_array_stream;

    test_file test;
    test_sequence<> seq;

    //--------------indirect_array_istream------------------------------------//
    {
        indirect_array_istream first(&seq[0], &seq[0] + seq.size());
        ifstream second(test.name().c_str());
        HPX_TEST_MSG(compare_streams_in_chars(first, second),
            "failed reading from indirect_array_istream in chars");
    }

    {
        indirect_array_istream first(&seq[0], &seq[0] + seq.size());
        ifstream second(test.name().c_str());
        HPX_TEST_MSG(compare_streams_in_chunks(first, second),
            "failed reading from indirect_array_istream in chunks");
    }

    //--------------indirect_array_ostream------------------------------------//
    {
        vector<char> dest(data_reps * data_length(), '?');
        indirect_array_ostream out(&dest[0], &dest[0] + dest.size());
        write_data_in_chars(out);
        HPX_TEST_MSG(std::equal(seq.begin(), seq.end(), dest.begin()),
            "failed writing to indirect_array_ostream in chunks");
    }

    {
        vector<char> dest(data_reps * data_length(), '?');
        indirect_array_ostream out(&dest[0], &dest[0] + dest.size());
        write_data_in_chunks(out);
        HPX_TEST_MSG(std::equal(seq.begin(), seq.end(), dest.begin()),
            "failed writing to indirect_array_ostream in chunks");
    }

    //--------------indirect_array_stream-------------------------------------//
    {
        vector<char> test(data_reps * data_length(), '?');
        indirect_array_stream io(&test[0], &test[0] + test.size());
        HPX_TEST_MSG(test_seekable_in_chars(io),
            "failed seeking within indirect_array_stream, in chars");
    }

    {
        vector<char> test(data_reps * data_length(), '?');
        indirect_array_stream io(&test[0], &test[0] + test.size());
        HPX_TEST_MSG(test_seekable_in_chunks(io),
            "failed seeking within indirect_array_stream, in chunks");
    }
}

int main(int, char*[])
{
    direct_adapter_test();
    return hpx::util::report_errors();
}
