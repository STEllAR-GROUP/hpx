//  Copyright (c) 2026 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// (C) Copyright 2008 CodeRage, LLC (turkanis at coderage dot com)
// (C) Copyright 2003-2007 Jonathan Turkanis

// See http://www.boost.org/libs/iostreams for documentation.

#include <hpx/config.hpp>
#include <hpx/iostreams/filter/bzip2.hpp>

// Julian Seward's "bzip.h" header. To configure HPX to work with libbz2, see
// the installation instructions here:
// http://boost.org/libs/iostreams/doc/index.html?path=7

#include "bzlib.h"

#include <iosfwd>

namespace hpx::iostreams {

    namespace bzip2 {

        // Status codes
        int const ok = BZ_OK;
        int const run_ok = BZ_RUN_OK;
        int const flush_ok = BZ_FLUSH_OK;
        int const finish_ok = BZ_FINISH_OK;
        int const stream_end = BZ_STREAM_END;
        int const sequence_error = BZ_SEQUENCE_ERROR;
        int const param_error = BZ_PARAM_ERROR;
        int const mem_error = BZ_MEM_ERROR;
        int const data_error = BZ_DATA_ERROR;
        int const data_error_magic = BZ_DATA_ERROR_MAGIC;
        int const io_error = BZ_IO_ERROR;
        int const unexpected_eof = BZ_UNEXPECTED_EOF;
        int const outbuff_full = BZ_OUTBUFF_FULL;
        int const config_error = BZ_CONFIG_ERROR;

        // Action codes
        int const finish = BZ_FINISH;
        int const run = BZ_RUN;
    }    // End namespace bzip2.

    //------------------Implementation of bzip2_error-----------------------------//
    bzip2_error::bzip2_error(int const error)
      : std::ios_base::failure("bzip2 error")
      , error_(error)
    {
    }

    void bzip2_error::check(int const error)
    {
        switch (error)
        {
        case BZ_OK:
        case BZ_RUN_OK:
        case BZ_FLUSH_OK:
        case BZ_FINISH_OK:
        case BZ_STREAM_END:
            return;
        case BZ_MEM_ERROR:
            throw std::bad_alloc();
        default:
            throw bzip2_error(error);
        }
    }

    //------------------Implementation of bzip2_base------------------------------//
    namespace detail {

        bzip2_base::bzip2_base(bzip2_params const& params)
          : params_(params)
          , stream_(new bz_stream)
          , ready_(false)
        {
        }

        bzip2_base::~bzip2_base()
        {
            delete static_cast<bz_stream*>(stream_);
        }

        void bzip2_base::before(char const*& src_begin, char const* src_end,
            char*& dest_begin, char const* dest_end) const
        {
            auto* s = static_cast<bz_stream*>(stream_);
            s->next_in = const_cast<char*>(src_begin);
            s->avail_in = static_cast<unsigned>(src_end - src_begin);
            s->next_out = dest_begin;
            s->avail_out = static_cast<unsigned>(dest_end - dest_begin);
        }

        void bzip2_base::after(char const*& src_begin, char*& dest_begin) const
        {
            auto* s = static_cast<bz_stream*>(stream_);
            src_begin = const_cast<char*>(s->next_in);
            dest_begin = s->next_out;
        }

        int bzip2_base::check_end(
            char const* src_begin, char const* dest_begin) const
        {
            auto* s = static_cast<bz_stream*>(stream_);
            if (src_begin == s->next_in && s->avail_in == 0 &&
                dest_begin == s->next_out)
            {
                return bzip2::unexpected_eof;
            }
            return bzip2::ok;
        }

        int bzip2_base::end(bool const compress, std::nothrow_t)
        {
            if (!ready_)
                return bzip2::ok;

            ready_ = false;
            auto* s = static_cast<bz_stream*>(stream_);
            return compress ? BZ2_bzCompressEnd(s) : BZ2_bzDecompressEnd(s);
        }

        void bzip2_base::end(bool compress)
        {
            bzip2_error::check(end(compress, std::nothrow));
        }

        int bzip2_base::compress(int const action) const
        {
            return BZ2_bzCompress(static_cast<bz_stream*>(stream_), action);
        }

        int bzip2_base::decompress() const
        {
            return BZ2_bzDecompress(static_cast<bz_stream*>(stream_));
        }

        void bzip2_base::do_init(bool const compress, bzip2::alloc_func,
            bzip2::free_func, void* derived)
        {
            auto* s = static_cast<bz_stream*>(stream_);

            // Current interface for customizing memory management is
            // non-conforming and has been disabled:
            //
            //    s->bzalloc = alloc;
            //    s->bzfree = free;
            s->bzalloc = nullptr;
            s->bzfree = nullptr;
            s->opaque = derived;
            bzip2_error::check(compress ?
                    BZ2_bzCompressInit(
                        s, params_.block_size, 0, params_.work_factor) :
                    BZ2_bzDecompressInit(s, 0, params_.small_));
            ready_ = true;
        }
    }    // namespace detail
}    // namespace hpx::iostreams
