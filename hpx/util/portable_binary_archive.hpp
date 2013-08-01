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
        disable_array_optimization  = 0x00010000,
        disable_data_chunking       = 0x00020000,

        all_archive_flags           = 0x0003e000    // all of the above
    };

    inline void
    reverse_bytes(char size, char* address)
    {
        std::reverse(address, address + size);
    }

    ///////////////////////////////////////////////////////////////////////
    union chunk_data
    {
        std::size_t index_;     // position inside the data buffer
        void const* pos_;       // pointer to external data buffer
    };

    enum chunk_type
    {
        chunk_type_index = 0,
        chunk_type_pointer = 1
    };

    struct chunk
    {
        chunk_type type_;
        chunk_data data_;       // index or pointer
        std::size_t size_;      // size of the chunk starting at index_/pos_
    };

    ///////////////////////////////////////////////////////////////////////
    inline chunk create_index_chunk(std::size_t index, std::size_t size)
    {
        chunk retval = { chunk_type_index, 0, size };
        retval.data_.index_ = index;
        return retval;
    }

    inline chunk create_pointer_chunk(void const* pos, std::size_t size)
    {
        chunk retval = { chunk_type_pointer, 0, size };
        retval.data_.pos_ = pos;
        return retval;
    }
}}

#endif // PORTABLE_BINARY_ARCHIVE_HPP
