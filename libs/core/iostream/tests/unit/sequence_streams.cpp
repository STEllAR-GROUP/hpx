//  Copyright (c) 2026 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// (C) Copyright 2008 CodeRage, LLC (turkanis at coderage dot com)
// (C) Copyright 2007 Jonathan Turkanis

// See http://www.boost.org/libs/iostreams for documentation.

#include <hpx/hpx_main.hpp>

#include "detail/read_input_seq_test.hpp"
#include "detail/read_seekable_seq_test.hpp"
#include "detail/write_output_seq_test.hpp"
#include "detail/write_seekable_seq_test.hpp"

int main(int, char*[])
{
    read_input_sequence_test();
    read_seekable_sequence_test();
    write_output_sequence_test();
    write_seekable_sequence_test();

    return hpx::util::report_errors();
}
