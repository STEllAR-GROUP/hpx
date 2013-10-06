// Copyright (c) 2007-2012 Hartmut Kaiser
// (C) Copyright 2002 Robert Ramey - http://www.rrsd.com .
//
// Use, modification and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#ifndef BASIC_BINARY_IPRIMITIVE_IMPL_HPP
#define BASIC_BINARY_IPRIMITIVE_IMPL_HPP

// MS compatible compilers support #pragma once
#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif

#include <hpx/util/basic_binary_iprimitive.hpp>

#include <boost/throw_exception.hpp>
#include <boost/archive/archive_exception.hpp>

namespace hpx { namespace util
{
    template <typename Archive>
    void basic_binary_iprimitive<Archive>::init(unsigned flags)
    {
        if (flags & boost::archive::no_header)
            return;

        // Detect  attempts to pass native binary archives across
        // incompatible platforms. This is not fool proof but its
        // better than nothing.
        unsigned char size;
        This()->load(size);
        if (sizeof(int) != size) {
            BOOST_THROW_EXCEPTION(
                boost::archive::archive_exception(
                    boost::archive::archive_exception::incompatible_native_format,
                    "size of int"
                )
            );
        }
        This()->load(size);
        if (sizeof(long) != size) {
            BOOST_THROW_EXCEPTION(
                boost::archive::archive_exception(
                    boost::archive::archive_exception::incompatible_native_format,
                    "size of long"
                )
            );
        }
        This()->load(size);
        if (sizeof(float) != size) {
            BOOST_THROW_EXCEPTION(
                boost::archive::archive_exception(
                    boost::archive::archive_exception::incompatible_native_format,
                    "size of float"
                )
            );
        }
        This()->load(size);
        if (sizeof(double) != size) {
            BOOST_THROW_EXCEPTION(
                boost::archive::archive_exception(
                    boost::archive::archive_exception::incompatible_native_format,
                    "size of double"
                )
            );
        }

        // for checking endian
        int i;
        This()->load(i);
        if (1 != i) {
            BOOST_THROW_EXCEPTION(
                boost::archive::archive_exception(
                    boost::archive::archive_exception::incompatible_native_format,
                    "endian setting"
                )
            );
        }
    }
}}

#endif

