//  Copyright (c) 2026 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// (C) Copyright 2011 Steven Watanabe

// See http://www.boost.org/libs/iostreams for documentation.

#include <hpx/modules/iostreams.hpp>
#include <hpx/modules/testing.hpp>

#include <fstream>

#include "detail/filters.hpp"
#include "detail/sequence.hpp"
#include "detail/temp_file.hpp"
#include "detail/verification.hpp"

void test_filtering_ostream_flush()
{
    using namespace std;
    using namespace hpx::iostreams;
    using namespace hpx::iostreams::test;

    lowercase_file lower;

    {
        temp_file dest;
        filtering_ostream out;
        out.push(tolower_filter<output>());
        out.push(file_sink(dest.name(), out_mode));
        write_data_in_chars(out);
        out.flush();
        HPX_TEST_MSG(compare_files(dest.name(), lower.name()),
            "failed writing to a filtering_ostream in chars with an "
            "output filter");
    }
}
