//  Copyright (c) 2026 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// (C) Copyright 2008 CodeRage, LLC (turkanis at coderage dot com)
// (C) Copyright 2004-2007 Jonathan Turkanis

// See http://www.boost.org/libs/iostreams/ for documentation.

#pragma once

#include <hpx/modules/iostream.hpp>
#include <hpx/modules/testing.hpp>

#include <istream>

#include "constants.hpp"
#include "temp_file.hpp"

using hpx::iostream::test::chunk_size;

bool putback_test_one(std::istream& is)
{
    try
    {
        do
        {
            char buf[chunk_size];
            is.read(buf, chunk_size);
            if (is.gcount() < static_cast<std::streamsize>(chunk_size))
                break;
            is.putback('a');
            if (is.get() != 'a')
                return false;
        } while (!is.eof());
        return true;
    }
    catch (std::exception&)
    {
        return false;
    }
}

bool putback_test_two(std::istream& is)
{
    try
    {
        do
        {
            char buf[chunk_size];
            is.read(buf, chunk_size);
            if (is.gcount() < static_cast<std::streamsize>(chunk_size))
                break;
            is.putback('a');
            is.putback('b');
            is.putback('c');
            is.putback('d');
            if (is.get() != 'd' || is.get() != 'c' || is.get() != 'b' ||
                is.get() != 'a')
            {
                return false;
            }
        } while (!is.eof());
        return true;
    }
    catch (std::exception&)
    {
        return false;
    }
}

template <typename Source>
bool putback_test_three(Source& src)
{
    try
    {
        while (true)
        {
            char buf[chunk_size];
            if (hpx::iostream::read(src, buf, chunk_size) < chunk_size)
                break;
            hpx::iostream::putback(src, 'a');
            if (hpx::iostream::get(src) != 'a')
                return false;
        }
        return true;
    }
    catch (std::exception&)
    {
        return false;
    }
}

template <typename Source>
bool putback_test_four(Source& src)
{
    try
    {
        while (true)
        {
            char buf[chunk_size];
            if (hpx::iostream::read(src, buf, chunk_size) < chunk_size)
                break;

            hpx::iostream::putback(src, 'a');
            hpx::iostream::putback(src, 'b');
            hpx::iostream::putback(src, 'c');
            hpx::iostream::putback(src, 'd');
            if (hpx::iostream::get(src) != 'd' ||
                hpx::iostream::get(src) != 'c' ||
                hpx::iostream::get(src) != 'b' ||
                hpx::iostream::get(src) != 'a')
            {
                return false;
            }
        }
        return true;
    }
    catch (std::exception&)
    {
        return false;
    }
}

void putback_test()
{
    using namespace std;
    using namespace hpx::iostream;
    using namespace hpx::iostream::test;

    test_file test;

    {
        filtering_istream is;
        is.set_device_buffer_size(0);
        is.push(file_source(test.name()));
        HPX_TEST_MSG(putback_test_one(is),
            "failed putting back to unbuffered filtering_istream");
    }

    {
        filtering_istream is;
        is.set_pback_size(4);
        is.push(file_source(test.name()));
        HPX_TEST_MSG(putback_test_two(is),
            "failed putting back to buffered filtering_istream");
    }

    {
        filtering_istream is;
        is.set_device_buffer_size(0);
        is.push(file_source(test.name()));
        HPX_TEST_MSG(putback_test_three(is),
            "failed putting back to unbuffered filtering_istream");
    }

    {
        filtering_istream is;
        is.set_pback_size(4);
        is.push(file_source(test.name()));
        HPX_TEST_MSG(putback_test_four(is),
            "failed putting back to buffered filtering_istream");
    }

    {
        filtering_istreambuf sb;
        sb.set_device_buffer_size(0);
        sb.push(file_source(test.name()));
        HPX_TEST_MSG(putback_test_three(sb),
            "failed putting back to unbuffered filtering_istream");
    }

    {
        filtering_istreambuf sb;
        sb.set_pback_size(4);
        sb.push(file_source(test.name()));
        HPX_TEST_MSG(putback_test_four(sb),
            "failed putting back to buffered filtering_istream");
    }
}
