//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>

#if defined(HPX_HAVE_BZIP2_COMPRESSION)
#include <hpx/runtime/actions/action_support.hpp>
#include <hpx/runtime/actions/compression/bzip2_serialization_filter.hpp>
#include <hpx/runtime/actions/guid_initialization.hpp>
#include <hpx/util/void_cast.hpp>

#include <boost/format.hpp>

///////////////////////////////////////////////////////////////////////////////
HPX_SERIALIZATION_REGISTER_TYPE_DEFINITION(hpx::actions::bzip2_serialization_filter);
HPX_REGISTER_BASE_HELPER(hpx::actions::bzip2_serialization_filter,
    bzip2_serialization_filter);

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace actions
{
    namespace detail
    {
        bzip2_compdecomp::bzip2_compdecomp()
          : boost::iostreams::detail::bzip2_base(
                boost::iostreams::bzip2_params(
                    boost::iostreams::bzip2::default_small)),
            compress_(false), eof_(false)
        {}

        bzip2_compdecomp::bzip2_compdecomp(bool compress,
                boost::iostreams::bzip2_params const& params)
          : boost::iostreams::detail::bzip2_base(params),
            compress_(compress), eof_(false)
        {}

        bzip2_compdecomp::~bzip2_compdecomp()
        {
            close();
        }

        bool bzip2_compdecomp::save(char const*& src_begin, char const* src_end,
            char*& dest_begin, char* dest_end, bool flush)
        {
            using namespace boost::iostreams;

            if (!this->ready())
                init();
            if (eof_)
                return false;

            this->before(src_begin, src_end, dest_begin, dest_end);
            int result = this->compress(flush ? bzip2::finish : bzip2::run);
            this->after(src_begin, dest_begin);
            (bzip2_error::check)(result);
            return !(eof_ = result == bzip2::stream_end);
        }

        bool bzip2_compdecomp::load(char const*& src_begin, char const* src_end,
            char*& dest_begin, char* dest_end)
        {
            using namespace boost::iostreams;

            if (eof_) {
                // reset the stream if there are more characters
                if(src_begin == src_end)
                    return false;
                else
                    this->close();
            }
            if (!this->ready())
                init();

            this->before(src_begin, src_end, dest_begin, dest_end);
            int result = this->decompress();
            if(result == bzip2::ok)
                result = this->check_end(src_begin, dest_begin);
            this->after(src_begin, dest_begin);
            (bzip2_error::check)(result);
            eof_ = result == bzip2::stream_end;
            return true;
        }

        void bzip2_compdecomp::close()
        {
            try {
                this->end(compress_);
            } catch (...) {
                eof_ = false;
                throw;
            }
            eof_ = false;
        }
    }

    bzip2_serialization_filter::~bzip2_serialization_filter()
    {
        detail::guid_initialization<bzip2_serialization_filter>();
    }

    void bzip2_serialization_filter::register_base()
    {
        util::void_cast_register_nonvirt<
            bzip2_serialization_filter, util::binary_filter>();
    }

    void bzip2_serialization_filter::set_max_compression_length(std::size_t size)
    {
        if (immediate_)
            buffer_.reserve(size);
    }

    void bzip2_serialization_filter::init_decompression_data(char const* buffer,
        std::size_t size, std::size_t decompressed_size)
    {
        if (immediate_) {
            buffer_.resize(decompressed_size);
            std::size_t s = load_impl(buffer_.data(), decompressed_size, buffer, size);
            if (s != size)
            {
                HPX_THROW_EXCEPTION(serialization_error,
                    "bzip2_serialization_filter::load",
                    boost::str(boost::format("decompression failure, number of "
                        "bytes expected: %d, number of bytes decoded: %d") %
                            size % s) );
            }
            current_ = 0;
        }
    }

    std::size_t bzip2_serialization_filter::load_impl(void* dst,
        std::size_t dst_count, void const* src, std::size_t src_count)
    {
        char const* src_begin = static_cast<char const*>(src);
        char* dst_begin = static_cast<char*>(dst);
        compdecomp_.load(src_begin, src_begin+src_count, dst_begin,
            dst_begin+dst_count);

        if (!immediate_ && 
            std::size_t(dst_begin-static_cast<char*>(dst)) != dst_count)
        {
            HPX_THROW_EXCEPTION(serialization_error,
                "bzip2_serialization_filter::load",
                boost::str(boost::format("decompression failure, number of "
                    "bytes expected: %d, number of bytes decoded: %d") %
                        dst_count % (dst_begin-static_cast<char*>(dst)) ));
            return 0;
        }
        return src_begin-static_cast<char const*>(src);
    }

    ///////////////////////////////////////////////////////////////////////////
    std::size_t bzip2_serialization_filter::load(void* dst,
        std::size_t dst_count, void const* src, std::size_t src_count)
    {
        if (!immediate_) 
            return load_impl(dst, dst_count, src, src_count);

        if (current_+dst_count > buffer_.size()) 
        {
            BOOST_THROW_EXCEPTION(
                boost::archive::archive_exception(
                    boost::archive::archive_exception::input_stream_error,
                    "archive data bstream is too short"));
        }
        std::memcpy(dst, &buffer_[current_], dst_count);
        current_ += dst_count;
        return 0;
    }

    ///////////////////////////////////////////////////////////////////////////
    std::size_t bzip2_serialization_filter::save(void* dst,
        std::size_t dst_count, void const* src, std::size_t src_count)
    {
        char const* src_begin = static_cast<char const*>(src);
        if (!immediate_) {
            char* dst_begin = static_cast<char*>(dst);
            compdecomp_.save(src_begin, src_begin+src_count, dst_begin,
                dst_begin+dst_count);
            return dst_begin-static_cast<char*>(dst);
        }

        std::copy(src_begin, src_begin+src_count, std::back_inserter(buffer_));
        return 0;       // no output written
    }

    ///////////////////////////////////////////////////////////////////////////
    std::size_t bzip2_serialization_filter::flush(void* dst,
        std::size_t dst_count)
    {
        char* dst_begin = static_cast<char*>(dst);
        if (!immediate_) {
            // flush the internal buffers
            char dummy = '\0';
            char const* src_begin = &dummy;
            char* dst_begin = static_cast<char*>(dst);
            if (compdecomp_.save(src_begin, src_begin, dst_begin,
                    dst_begin+dst_count, true))
            {
                HPX_THROW_EXCEPTION(serialization_error,
                    "bzip2_serialization_filter::flush",
                    "compression failure, flushing did not reach end of data");
                return 0;
            }
        }
        else {
            // compress everything in one go
            char const* src_begin = buffer_.data();
            if (compdecomp_.save(src_begin, src_begin+buffer_.size(),
                    dst_begin, dst_begin+dst_count, true))
            {
                HPX_THROW_EXCEPTION(serialization_error,
                    "bzip2_serialization_filter::flush",
                    "compression failure, flushing did not reach end of data");
                return 0;
            }
        }
        return dst_begin-static_cast<char*>(dst);
    }
}}

#endif
