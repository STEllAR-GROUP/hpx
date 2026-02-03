//  Copyright (c) 2026 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// (C) Copyright 2008 CodeRage, LLC (turkanis at coderage dot com)
// (C) Copyright 2003-2007 Jonathan Turkanis
//
// See http://www.boost.org/libs/iostreams for documentation.

// Contains category and mode tags for classifying filters, devices and
// standard stream and stream buffers types.

#pragma once

#include <hpx/config.hpp>

namespace hpx::iostream {

    //------------------Tags for dispatch according to i/o mode-------------------//
    HPX_CXX_CORE_EXPORT struct any_tag
    {
    };

    namespace detail {

        HPX_CXX_CORE_EXPORT struct two_sequence : virtual any_tag
        {
        };

        HPX_CXX_CORE_EXPORT struct random_access : virtual any_tag
        {
        };

        HPX_CXX_CORE_EXPORT struct one_head : virtual any_tag
        {
        };

        HPX_CXX_CORE_EXPORT struct two_head : virtual any_tag
        {
        };
    }    // namespace detail

    HPX_CXX_CORE_EXPORT struct input : virtual any_tag
    {
    };

    HPX_CXX_CORE_EXPORT struct output : virtual any_tag
    {
    };

    HPX_CXX_CORE_EXPORT struct bidirectional
      : virtual input
      , virtual output
      , detail::two_sequence
    {
    };

    // Pseudo-mode.
    HPX_CXX_CORE_EXPORT struct dual_use
      : virtual input
      , virtual output
    {
    };

    HPX_CXX_CORE_EXPORT struct input_seekable
      : virtual input
      , virtual detail::random_access
    {
    };

    HPX_CXX_CORE_EXPORT struct output_seekable
      : virtual output
      , virtual detail::random_access
    {
    };

    HPX_CXX_CORE_EXPORT struct seekable
      : virtual input_seekable
      , virtual output_seekable
      , detail::one_head
    {
    };

    HPX_CXX_CORE_EXPORT struct dual_seekable
      : virtual input_seekable
      , virtual output_seekable
      , detail::two_head
    {
    };

    HPX_CXX_CORE_EXPORT struct bidirectional_seekable
      : input_seekable
      , output_seekable
      , bidirectional
      , detail::two_head
    {
    };

    //------------------Tags for use as i/o categories----------------------------//
    HPX_CXX_CORE_EXPORT struct device_tag : virtual any_tag
    {
    };

    HPX_CXX_CORE_EXPORT struct filter_tag : virtual any_tag
    {
    };

    //
    // Tags for optional behavior.
    //
    HPX_CXX_CORE_EXPORT struct peekable_tag : virtual any_tag
    {
    };

    HPX_CXX_CORE_EXPORT struct closable_tag : virtual any_tag
    {
    };

    HPX_CXX_CORE_EXPORT struct flushable_tag : virtual any_tag
    {
    };

    HPX_CXX_CORE_EXPORT struct localizable_tag : virtual any_tag
    {
    };

    HPX_CXX_CORE_EXPORT struct optimally_buffered_tag : virtual any_tag
    {
    };

    HPX_CXX_CORE_EXPORT struct direct_tag : virtual any_tag
    {
    };

    HPX_CXX_CORE_EXPORT struct multichar_tag : virtual any_tag
    {
    };

    HPX_CXX_CORE_EXPORT struct source_tag
      : device_tag
      , input
    {
    };

    HPX_CXX_CORE_EXPORT struct sink_tag
      : device_tag
      , output
    {
    };

    HPX_CXX_CORE_EXPORT struct bidirectional_device_tag
      : device_tag
      , bidirectional
    {
    };

    HPX_CXX_CORE_EXPORT struct seekable_device_tag
      : virtual device_tag
      , seekable
    {
    };

    HPX_CXX_CORE_EXPORT struct input_filter_tag
      : filter_tag
      , input
    {
    };

    HPX_CXX_CORE_EXPORT struct output_filter_tag
      : filter_tag
      , output
    {
    };

    HPX_CXX_CORE_EXPORT struct bidirectional_filter_tag
      : filter_tag
      , bidirectional
    {
    };

    HPX_CXX_CORE_EXPORT struct seekable_filter_tag
      : filter_tag
      , seekable
    {
    };

    HPX_CXX_CORE_EXPORT struct dual_use_filter_tag
      : filter_tag
      , dual_use
    {
    };

    HPX_CXX_CORE_EXPORT struct multichar_input_filter_tag
      : multichar_tag
      , input_filter_tag
    {
    };

    HPX_CXX_CORE_EXPORT struct multichar_output_filter_tag
      : multichar_tag
      , output_filter_tag
    {
    };

    HPX_CXX_CORE_EXPORT struct multichar_bidirectional_filter_tag
      : multichar_tag
      , bidirectional_filter_tag
    {
    };

    HPX_CXX_CORE_EXPORT struct multichar_seekable_filter_tag
      : multichar_tag
      , seekable_filter_tag
    {
    };

    HPX_CXX_CORE_EXPORT struct multichar_dual_use_filter_tag
      : multichar_tag
      , dual_use_filter_tag
    {
    };

    //
    // Tags for standard streams and streambufs.
    //
    HPX_CXX_CORE_EXPORT struct std_io_tag : virtual localizable_tag
    {
    };

    HPX_CXX_CORE_EXPORT struct istream_tag
      : virtual device_tag
      , virtual peekable_tag
      , virtual std_io_tag
    {
    };

    HPX_CXX_CORE_EXPORT struct ostream_tag
      : virtual device_tag
      , virtual std_io_tag
    {
    };

    HPX_CXX_CORE_EXPORT struct iostream_tag
      : istream_tag
      , ostream_tag
    {
    };

    HPX_CXX_CORE_EXPORT struct streambuf_tag
      : device_tag
      , peekable_tag
      , std_io_tag
    {
    };

    HPX_CXX_CORE_EXPORT struct ifstream_tag
      : input_seekable
      , closable_tag
      , istream_tag
    {
    };

    HPX_CXX_CORE_EXPORT struct ofstream_tag
      : output_seekable
      , closable_tag
      , ostream_tag
    {
    };

    HPX_CXX_CORE_EXPORT struct fstream_tag
      : seekable
      , closable_tag
      , iostream_tag
    {
    };

    HPX_CXX_CORE_EXPORT struct filebuf_tag
      : seekable
      , closable_tag
      , streambuf_tag
    {
    };

    HPX_CXX_CORE_EXPORT struct istringstream_tag
      : input_seekable
      , istream_tag
    {
    };

    HPX_CXX_CORE_EXPORT struct ostringstream_tag
      : output_seekable
      , ostream_tag
    {
    };

    HPX_CXX_CORE_EXPORT struct stringstream_tag
      : dual_seekable
      , iostream_tag
    {
    };

    HPX_CXX_CORE_EXPORT struct stringbuf_tag
      : dual_seekable
      , streambuf_tag
    {
    };

    HPX_CXX_CORE_EXPORT struct generic_istream_tag
      : input_seekable
      , istream_tag
    {
    };

    HPX_CXX_CORE_EXPORT struct generic_ostream_tag
      : output_seekable
      , ostream_tag
    {
    };

    HPX_CXX_CORE_EXPORT struct generic_iostream_tag
      : seekable
      , iostream_tag
    {
    };

    HPX_CXX_CORE_EXPORT struct generic_streambuf_tag
      : seekable
      , streambuf_tag
    {
    };
}    // namespace hpx::iostream
