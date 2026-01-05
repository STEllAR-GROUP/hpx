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

#include <cctype>
#include <string>

#include "detail/filter_tests.hpp"
#include "detail/filters.hpp"

using namespace hpx::iostreams;
using namespace hpx::iostreams::test;

std::string const lower =
    "in addition to providing an abstract framework the "
    "library provides a number of concrete filters, sources "
    "and sinks which serve as example applications of the "
    "library but are also useful in their own right. these "
    "include components for accessing memory-mapped files, "
    "for file access via operating system file descriptors, "
    "for code conversion, for text filtering with regular "
    "expressions, for line-ending conversion and for "
    "compression and decompression in the zlib, gzip and "
    "bzip2 formats.";

std::string const upper =
    "IN ADDITION TO PROVIDING AN ABSTRACT FRAMEWORK THE "
    "LIBRARY PROVIDES A NUMBER OF CONCRETE FILTERS, SOURCES "
    "AND SINKS WHICH SERVE AS EXAMPLE APPLICATIONS OF THE "
    "LIBRARY BUT ARE ALSO USEFUL IN THEIR OWN RIGHT. THESE "
    "INCLUDE COMPONENTS FOR ACCESSING MEMORY-MAPPED FILES, "
    "FOR FILE ACCESS VIA OPERATING SYSTEM FILE DESCRIPTORS, "
    "FOR CODE CONVERSION, FOR TEXT FILTERING WITH REGULAR "
    "EXPRESSIONS, FOR LINE-ENDING CONVERSION AND FOR "
    "COMPRESSION AND DECOMPRESSION IN THE ZLIB, GZIP AND "
    "BZIP2 FORMATS.";

struct toupper_dual_use_filter : public dual_use_filter
{
    template <typename Source>
    int get(Source& s)
    {
        int c = hpx::iostreams::get(s);
        return c != EOF && c != WOULD_BLOCK ? std::toupper((unsigned char) c) :
                                              c;
    }

    template <typename Sink>
    bool put(Sink& s, char c)
    {
        return hpx::iostreams::put(s, (char) std::toupper((unsigned char) c));
    }
};

struct tolower_dual_use_filter : public dual_use_filter
{
    template <typename Source>
    int get(Source& s)
    {
        int c = hpx::iostreams::get(s);
        return c != EOF && c != WOULD_BLOCK ? std::tolower((unsigned char) c) :
                                              c;
    }

    template <typename Sink>
    bool put(Sink& s, char c)
    {
        return hpx::iostreams::put(s, (char) std::tolower((unsigned char) c));
    }
};

void filter_test()
{
    HPX_TEST(test_input_filter(toupper_filter<input>(), lower, upper));
    HPX_TEST(
        test_input_filter(toupper_multichar_filter<input>(), lower, upper));
    HPX_TEST(test_input_filter(toupper_dual_use_filter(), lower, upper));
    HPX_TEST(test_output_filter(tolower_filter<output>(), upper, lower));
    HPX_TEST(
        test_output_filter(tolower_multichar_filter<output>(), upper, lower));
    HPX_TEST(test_output_filter(tolower_dual_use_filter(), upper, lower));
    HPX_TEST(test_filter_pair(
        tolower_filter<output>(), toupper_filter<input>(), upper));
    HPX_TEST(test_filter_pair(tolower_multichar_filter<output>(),
        toupper_multichar_filter<input>(), upper));
    HPX_TEST(test_filter_pair(
        tolower_dual_use_filter(), toupper_dual_use_filter(), upper));
}

int main(int, char*[])
{
    filter_test();
    return hpx::util::report_errors();
}
