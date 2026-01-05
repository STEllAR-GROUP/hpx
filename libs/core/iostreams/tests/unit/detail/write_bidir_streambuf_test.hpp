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

void write_bidirectional_streambuf_test()
{
    using namespace std;
    using namespace hpx::iostreams;
    using namespace hpx::iostreams::test;

    test_file test;

    {
        temp_file test2;
        {
            filebuf dest;
            dest.open(test2.name().c_str(), out_mode);
            filtering_stream<bidirectional> out(dest, 0);
            write_data_in_chars(out);
        }
        HPX_TEST_MSG(compare_files(test2.name(), test.name()),
            "failed writing to filtering_stream<bidirectional> based "
            "on a streambuf in chars with no buffer");
    }

    {
        temp_file test2;
        {
            filebuf dest;
            dest.open(test2.name().c_str(), out_mode);
            filtering_stream<bidirectional> out(dest, 0);
            write_data_in_chunks(out);
        }
        HPX_TEST_MSG(compare_files(test2.name(), test.name()),
            "failed writing to filtering_stream<bidirectional> based "
            "on a streambuf in chunks with no buffer");
    }

    {
        temp_file test2;
        {
            filebuf dest;
            dest.open(test2.name().c_str(), out_mode);
            filtering_stream<bidirectional> out(dest);
            write_data_in_chars(out);
        }
        HPX_TEST_MSG(compare_files(test2.name(), test.name()),
            "failed writing to filtering_stream<bidirectional> based "
            "on a streambuf in chars with large buffer");
    }

    {
        temp_file test2;
        {
            filebuf dest;
            dest.open(test2.name().c_str(), out_mode);
            filtering_stream<bidirectional> out(dest);
            write_data_in_chunks(out);
        }
        HPX_TEST_MSG(compare_files(test2.name(), test.name()),
            "failed writing to filtering_stream<bidirectional> based "
            "on a streambuf in chunks with large buffer");
    }
}
