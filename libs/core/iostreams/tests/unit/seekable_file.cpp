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

#include "detail/temp_file.hpp"
#include "detail/verification.hpp"

using namespace std;
using namespace hpx::iostreams;
using namespace hpx::iostreams::test;

void seekable_file_test()
{
    {
        temp_file temp;
        file f(temp.name(),
            std::ios_base::in | std::ios_base::out | std::ios_base::trunc |
                std::ios_base::binary);
        filtering_stream<seekable> io(f);
        io.exceptions(std::ios_base::failbit | std::ios_base::badbit);
        HPX_TEST_MSG(test_seekable_in_chars(io),
            "failed seeking within a file, in chars");
    }

    {
        temp_file temp;
        file f(temp.name(),
            std::ios_base::in | std::ios_base::out | std::ios_base::trunc |
                std::ios_base::binary);
        filtering_stream<seekable> io(f);
        io.exceptions(std::ios_base::failbit | std::ios_base::badbit);
        HPX_TEST_MSG(test_seekable_in_chunks(io),
            "failed seeking within a file, in chunks");
    }
}

int main(int, char*[])
{
    seekable_file_test();
    return hpx::util::report_errors();
}
