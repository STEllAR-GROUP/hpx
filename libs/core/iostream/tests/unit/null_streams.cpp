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
#include <hpx/modules/iostream.hpp>
#include <hpx/modules/testing.hpp>

#include <cctype>

#include "detail/temp_file.hpp"
#include "detail/verification.hpp"

using namespace std;
using namespace hpx::iostream;
using namespace hpx::iostream::test;

void read_null_source()
{
    stream<null_source> in;
    in.open(null_source());
    in.get();
    HPX_TEST(in.eof());
}

void write_null_sink()
{
    stream<null_sink> out;
    out.open(null_sink());
    write_data_in_chunks(out);
    HPX_TEST(out.good());
}

int main(int, char*[])
{
    read_null_source();
    write_null_sink();
    return hpx::util::report_errors();
}
