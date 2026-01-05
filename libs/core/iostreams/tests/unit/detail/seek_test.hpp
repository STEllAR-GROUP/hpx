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

#include <cstddef>
#include <string>

#include "detail/container_device.hpp"
#include "detail/verification.hpp"

void seek_test()
{
    using namespace std;
    using namespace hpx::iostreams;
    using namespace hpx::iostreams::test;

    {
        string test(data_reps * data_length(), '\0');
        filtering_stream<seekable> io;
        io.push(container_device<string>(test));
        HPX_TEST_MSG(test_seekable_in_chars(io),
            "failed seeking within a filtering_stream<seekable>, in chars");
    }

    {
        string test(data_reps * data_length(), '\0');
        filtering_stream<seekable> io;
        io.push(container_device<string>(test));
        HPX_TEST_MSG(test_seekable_in_chunks(io),
            "failed seeking within a filtering_stream<seekable>, in chunks");
    }
}
