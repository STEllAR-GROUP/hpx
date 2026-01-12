//  Copyright (c) 2026 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// (C) Copyright 2008 CodeRage, LLC (turkanis at coderage dot com)
// (C) Copyright 2003-2007 Jonathan Turkanis

// See http://www.boost.org/libs/iostreams for documentation.

// To configure Boost to work with zlib, see the installation instructions here:
// http://boost.org/libs/iostreams/doc/index.html?path=7

#include <hpx/config.hpp>
#include <hpx/iostream/filter/zlib.hpp>

// Jean-loup Gailly's and Mark Adler's "zlib.h" header. To configure Boost to
// work with zlib, see the installation instructions here:
// http://boost.org/libs/iostreams/doc/index.html?path=7
#include "zlib.h"

namespace hpx::iostream {

    namespace zlib {

        // Compression levels
        int const no_compression = Z_NO_COMPRESSION;
        int const best_speed = Z_BEST_SPEED;
        int const best_compression = Z_BEST_COMPRESSION;
        int const default_compression = Z_DEFAULT_COMPRESSION;

        // Compression methods
        int const deflated = Z_DEFLATED;

        // Compression strategies
        int const default_strategy = Z_DEFAULT_STRATEGY;
        int const filtered = Z_FILTERED;
        int const huffman_only = Z_HUFFMAN_ONLY;

        // Status codes
        int const okay = Z_OK;
        int const stream_end = Z_STREAM_END;
        int const stream_error = Z_STREAM_ERROR;
        int const version_error = Z_VERSION_ERROR;
        int const data_error = Z_DATA_ERROR;
        int const mem_error = Z_MEM_ERROR;
        int const buf_error = Z_BUF_ERROR;

        // Flush codes
        int const finish = Z_FINISH;
        int const no_flush = Z_NO_FLUSH;
        int const sync_flush = Z_SYNC_FLUSH;
    }    // End namespace zlib.

    //------------------Implementation of zlib_error------------------------------//
    zlib_error::zlib_error(int const error)
      : std::ios_base::failure("zlib error")
      , error_(error)
    {
    }

    void(zlib_error::check)(int const error)
    {
        switch (error)
        {
        case Z_OK:
        case Z_STREAM_END:
            //case Z_BUF_ERROR:
            return;
        case Z_MEM_ERROR:
            throw std::bad_alloc();
        default:
            throw zlib_error(error);
        }
    }

    //------------------Implementation of zlib_base-------------------------------//
    namespace detail {

        zlib_base::zlib_base()
          : stream_(new z_stream)
          , calculate_crc_(false)
          , crc_(0)
          , crc_imp_(0)
          , total_in_(0)
          , total_out_(0)
        {
        }

        zlib_base::~zlib_base()
        {
            delete static_cast<z_stream*>(stream_);
        }

        void zlib_base::before(char const*& src_begin, char const* src_end,
            char*& dest_begin, char const* dest_end) const
        {
            auto* s = static_cast<z_stream*>(stream_);
            s->next_in =
                reinterpret_cast<zlib::byte*>(const_cast<char*>(src_begin));
            s->avail_in = static_cast<zlib::uint>(src_end - src_begin);
            s->next_out = reinterpret_cast<zlib::byte*>(dest_begin);
            s->avail_out = static_cast<zlib::uint>(dest_end - dest_begin);
        }

        void zlib_base::after(
            char const*& src_begin, char*& dest_begin, bool const compress)
        {
            auto* s = static_cast<z_stream*>(stream_);
            auto* next_in = reinterpret_cast<char const*>(s->next_in);
            auto* next_out = reinterpret_cast<char*>(s->next_out);
            if (calculate_crc_)
            {
                zlib::byte const* buf = compress ?
                    reinterpret_cast<zlib::byte const*>(src_begin) :
                    reinterpret_cast<zlib::byte const*>(
                        const_cast<char const*>(dest_begin));
                zlib::uint const length = compress ?
                    static_cast<zlib::uint>(next_in - src_begin) :
                    static_cast<zlib::uint>(next_out - dest_begin);
                crc_ = crc_imp_ = crc32(crc_imp_, buf, length);
            }

            total_in_ = s->total_in;
            total_out_ = s->total_out;
            src_begin = next_in;
            dest_begin = next_out;
        }

        int zlib_base::xdeflate(int const flush) const
        {
            return ::deflate(static_cast<z_stream*>(stream_), flush);
        }

        int zlib_base::xinflate(int const flush) const
        {
            return ::inflate(static_cast<z_stream*>(stream_), flush);
        }

        void zlib_base::reset(bool const compress, bool const realloc)
        {
            auto* s = static_cast<z_stream*>(stream_);

            // Undiagnosed bug:
            // deflateReset(), etc., return Z_DATA_ERROR
            //zlib_error::check (
            realloc ? (compress ? deflateReset(s) : inflateReset(s)) :
                      (compress ? deflateEnd(s) : inflateEnd(s));
            //);
            crc_imp_ = 0;
        }

        void zlib_base::do_init(zlib_params const& p, bool const compress,
            zlib::xalloc_func, zlib::xfree_func, void* derived)
        {
            calculate_crc_ = p.calculate_crc;
            auto* s = static_cast<z_stream*>(stream_);

            // Current interface for customizing memory management
            // is non-conforming and has been disabled:
            //    s->zalloc = alloc;
            //    s->zfree = free;
            s->zalloc = nullptr;
            s->zfree = nullptr;
            s->opaque = derived;

            int const window_bits = p.noheader ? -p.window_bits : p.window_bits;
            (zlib_error::check)(compress ?
                    deflateInit2(s, p.level, p.method, window_bits, p.mem_level,
                        p.strategy) :
                    inflateInit2(s, window_bits));
        }
    }    // End namespace detail.
}    // namespace hpx::iostream
