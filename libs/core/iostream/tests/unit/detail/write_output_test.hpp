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

#include "sequence.hpp"
#include "temp_file.hpp"
#include "verification.hpp"

void write_output_test()
{
    using namespace std;
    using namespace hpx;
    using namespace hpx::iostream;
    using namespace hpx::iostream::test;

    test_file test;

    {
        temp_file test2;
        filtering_ostream out(file_sink(test2.name(), out_mode), 0);
        write_data_in_chars(out);
        out.reset();
        HPX_TEST_MSG(compare_files(test2.name(), test.name()),
            "failed writing to filtering_ostream in chars with no buffer");
    }

    {
        temp_file test2;
        filtering_ostream out(file_sink(test2.name(), out_mode), 0);
        write_data_in_chunks(out);
        out.reset();
        HPX_TEST_MSG(compare_files(test2.name(), test.name()),
            "failed writing to filtering_ostream in chunks with no buffer");
    }

    {
        temp_file test2;
        filtering_ostream out(file_sink(test2.name(), out_mode));
        write_data_in_chars(out);
        out.reset();
        HPX_TEST_MSG(compare_files(test2.name(), test.name()),
            "failed writing to filtering_ostream in chars with buffer");
    }

    {
        temp_file test2;
        filtering_ostream out(file_sink(test2.name(), out_mode));
        write_data_in_chunks(out);
        out.reset();
        HPX_TEST_MSG(compare_files(test2.name(), test.name()),
            "failed writing to filtering_ostream in chunks with buffer");
    }
}
