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

#include "detail/read_input_test.hpp"
#include "detail/read_bidir_test.hpp"
#include "detail/read_seekable_test.hpp"
#include "detail/read_bidir_streambuf_test.hpp"
#include "detail/read_input_istream_test.hpp"

#include "detail/write_output_test.hpp"
#include "detail/write_bidir_test.hpp"
#include "detail/write_seekable_test.hpp"
#include "detail/write_output_iterator_test.hpp"
#include "detail/write_bidir_streambuf_test.hpp"
#include "detail/write_output_ostream_test.hpp"

#include "detail/read_input_filter_test.hpp"
#include "detail/read_bidir_filter_test.hpp"
#include "detail/write_output_filter_test.hpp"
#include "detail/write_bidir_filter_test.hpp"

#include "detail/seek_test.hpp"
#include "detail/putback_test.hpp"
#include "detail/filtering_stream_flush_test.hpp"

int main(int, char* [])
{
    read_input_test();
    read_bidirectional_test();
    read_seekable_test();
    read_bidirectional_streambuf_test();
    read_input_istream_test();

    write_output_test();
    write_bidirectional_test();
    write_seekable_test();
    write_output_iterator_test();
    write_bidirectional_streambuf_test();
    write_output_ostream_test();

    read_input_filter_test();
    read_bidirectional_filter_test();
    write_output_filter_test();
    write_bidirectional_filter_test();

    seek_test();
    putback_test();
    test_filtering_ostream_flush();

    return hpx::util::report_errors();
}
