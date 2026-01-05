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

#include "detail/filters.hpp"
#include "detail/temp_file.hpp"
#include "detail/verification.hpp"

using namespace std;
using namespace hpx::iostreams;
using namespace hpx::iostreams::test;

void flush_test()
{
    {
        stream_buffer<null_sink> null;
        null.open(null_sink());
        HPX_TEST_MSG(
            hpx::iostreams::flush(null), "failed flushing stream_buffer");
        HPX_TEST_MSG(null.strict_sync(),
            "failed strict-syncing stream_buffer with "
            "non-flushable resource");
    }

    {
        stream<null_sink> null;
        null.open(null_sink());
        HPX_TEST_MSG(hpx::iostreams::flush(null), "failed flushing stream");
        HPX_TEST_MSG(null.strict_sync(),
            "failed strict-syncing stream with "
            "non-flushable resource");
    }

    {
        filtering_ostream null;
        null.push(null_sink());
        HPX_TEST_MSG(
            hpx::iostreams::flush(null), "failed flushing filtering_ostream");
        HPX_TEST_MSG(null.strict_sync(),
            "failed strict-syncing filtering_ostream with "
            "non-flushable resource");
    }

    {
        filtering_ostream null;
        null.push(tolower_filter<output>());
        null.push(null_sink());
        HPX_TEST_MSG(hpx::iostreams::flush(null),
            "failed flushing filtering_ostream with non-flushable filter");
        HPX_TEST_MSG(!null.strict_sync(),
            "strict-syncing filtering_ostream with "
            "non-flushable filter succeeded");
    }

    {
        vector<char> dest1;
        vector<char> dest2;
        filtering_ostream out;
        out.set_auto_close(false);
        out.push(flushable_output_filter());

        // Write to dest1.
        out.push(hpx::iostreams::back_inserter(dest1));
        write_data_in_chunks(out);
        out.flush();

        // Write to dest2.
        out.pop();
        out.push(hpx::iostreams::back_inserter(dest2));
        write_data_in_chunks(out);
        out.flush();

        HPX_TEST_MSG(dest1.size() == dest2.size() &&
                std::equal(dest1.begin(), dest1.end(), dest2.begin()),
            "failed flush filtering_ostream with auto_close disabled");
    }

    {
        vector<char> dest1;
        vector<char> dest2;
        filtering_ostream out;
        out.set_auto_close(false);
        out.push(flushable_output_filter());
        out.push(flushable_output_filter());

        // Write to dest1.
        out.push(hpx::iostreams::back_inserter(dest1));
        write_data_in_chunks(out);
        out.flush();

        // Write to dest2.
        out.pop();
        out.push(hpx::iostreams::back_inserter(dest2));
        write_data_in_chunks(out);
        out.flush();

        HPX_TEST_MSG(dest1.size() == dest2.size() &&
                std::equal(dest1.begin(), dest1.end(), dest2.begin()),
            "failed flush filtering_ostream with two flushable filters "
            "with auto_close disabled");
    }
}

int main(int, char*[])
{
    flush_test();
    return hpx::util::report_errors();
}
