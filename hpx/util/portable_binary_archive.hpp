#ifndef PORTABLE_BINARY_ARCHIVE_HPP
#define PORTABLE_BINARY_ARCHIVE_HPP

// (C) Copyright 2002 Robert Ramey - http://www.rrsd.com .
// Use, modification and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

// MS compatible compilers support #pragma once
#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif

#include <boost/config.hpp>
#include <boost/cstdint.hpp>
#include <boost/serialization/pfto.hpp>
#include <boost/static_assert.hpp>
#include <boost/archive/basic_archive.hpp>
#include <boost/detail/endian.hpp>

#include <algorithm>
#include <climits>
#if CHAR_BIT != 8
#  error This code assumes an eight-bit byte.
#endif

namespace hpx { namespace util
{
    enum portable_binary_archive_flags
    {
        enable_compression          = 0x00002000,
        endian_big                  = 0x00004000,
        endian_little               = 0x00008000,
        disable_array_optimization  = 0x00010000
    };

    inline void
    reverse_bytes(char size, char* address)
    {
        std::reverse(address, address + size);
    }
}}

#endif // PORTABLE_BINARY_ARCHIVE_HPP
