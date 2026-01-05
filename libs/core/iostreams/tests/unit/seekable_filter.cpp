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

#include <iosfwd>
#include <vector>

#include "detail/container_device.hpp"
#include "detail/filters.hpp"
#include "detail/temp_file.hpp"
#include "detail/verification.hpp"

using namespace std;
using namespace hpx::iostreams;
using namespace hpx::iostreams::test;

void seekable_filter_test()
{
    {
        vector<char> test(data_reps * data_length(), '0');
        filtering_stream<seekable> io;
        io.push(identity_seekable_filter());
        io.push(container_device<vector<char>>(test));
        io.exceptions(std::ios_base::failbit | std::ios_base::badbit);
        HPX_TEST_MSG(test_seekable_in_chars(io),
            "failed seeking within a file, in chars");
    }

    {
        vector<char> test(data_reps * data_length(), '0');
        filtering_stream<seekable> io;
        io.push(identity_seekable_filter());
        io.push(container_device<vector<char>>(test));
        io.exceptions(std::ios_base::failbit | std::ios_base::badbit);
        HPX_TEST_MSG(test_seekable_in_chunks(io),
            "failed seeking within a file, in chunks");
    }

    {
        vector<char> test(data_reps * data_length(), '0');
        filtering_stream<seekable> io;
        io.push(identity_seekable_multichar_filter());
        io.push(container_device<vector<char>>(test));
        io.exceptions(std::ios_base::failbit | std::ios_base::badbit);
        HPX_TEST_MSG(test_seekable_in_chars(io),
            "failed seeking within a file, in chars");
    }

    {
        vector<char> test(data_reps * data_length(), '0');
        filtering_stream<seekable> io;
        io.push(identity_seekable_multichar_filter());
        io.push(container_device<vector<char>>(test));
        io.exceptions(std::ios_base::failbit | std::ios_base::badbit);
        HPX_TEST_MSG(test_seekable_in_chunks(io),
            "failed seeking within a file, in chunks");
    }
}

int main(int, char*[])
{
    seekable_filter_test();
    return hpx::util::report_errors();
}
