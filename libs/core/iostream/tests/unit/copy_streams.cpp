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

#include <algorithm>
#include <vector>

#include "detail/container_device.hpp"
#include "detail/sequence.hpp"

using namespace std;
using namespace hpx;
using namespace hpx::iostream;
using namespace hpx::iostream::test;

//------------------Definition of stream types--------------------------------//
using vector_source = container_source<vector<char>>;
using vector_sink = container_sink<vector<char>>;
using vector_istream = stream<vector_source>;
using vector_ostream = stream<vector_sink>;

//------------------Definition of copy_test-----------------------------------//
void copy_test()
{
    // Stream to stream
    {
        test_sequence<> src;
        vector<char> dest;
        vector_istream first;
        vector_ostream second;
        first.open(vector_source(src));
        second.open(vector_sink(dest));
        HPX_TEST_MSG(hpx::iostream::copy(first, second) ==
                    static_cast<streamsize>(src.size()) &&
                src == dest,
            "failed copying from stream to stream");
    }

    // Stream to indirect sink
    {
        test_sequence<> src;
        vector<char> dest;
        vector_istream in;
        vector_sink out(dest);
        in.open(vector_source(src));
        HPX_TEST_MSG(hpx::iostream::copy(in, out) ==
                    static_cast<streamsize>(src.size()) &&
                src == dest,
            "failed copying from stream to indirect sink");
    }

    // Indirect source to stream
    {
        test_sequence<> src;
        vector<char> dest;
        vector_source in(src);
        vector_ostream out;
        out.open(vector_sink(dest));
        HPX_TEST_MSG(hpx::iostream::copy(in, out) ==
                    static_cast<streamsize>(src.size()) &&
                src == dest,
            "failed copying from indirect source to stream");
    }

    // Indirect source to indirect sink
    {
        test_sequence<> src;
        vector<char> dest;
        vector_source in(src);
        vector_sink out(dest);
        HPX_TEST_MSG(hpx::iostream::copy(in, out) ==
                    static_cast<streamsize>(src.size()) &&
                src == dest,
            "failed copying from indirect source to indirect sink");
    }

    // Direct source to direct sink
    {
        test_sequence<> src;
        vector<char> dest(src.size(), '?');
        array_source<char> in(&src[0], &src[0] + src.size());
        array_sink<char> out(&dest[0], &dest[0] + dest.size());
        HPX_TEST_MSG(hpx::iostream::copy(in, out) ==
                    static_cast<streamsize>(src.size()) &&
                src == dest,
            "failed copying from direct source to direct sink");
    }

    // Direct source to indirect sink
    {
        test_sequence<> src;
        vector<char> dest;
        array_source<char> in(&src[0], &src[0] + src.size());
        vector_ostream out(dest);
        HPX_TEST_MSG(hpx::iostream::copy(in, out) ==
                    static_cast<streamsize>(src.size()) &&
                src == dest,
            "failed copying from direct source to indirect sink");
    }

    // Indirect source to direct sink
    {
        test_sequence<> src;
        vector<char> dest(src.size(), '?');
        vector_istream in;
        array_sink<char> out(&dest[0], &dest[0] + dest.size());
        in.open(vector_source(src));
        HPX_TEST_MSG(hpx::iostream::copy(in, out) ==
                    static_cast<streamsize>(src.size()) &&
                src == dest,
            "failed copying from indirect source to direct sink");
    }
}

int main(int, char*[])
{
    copy_test();
    return hpx::util::report_errors();
}
