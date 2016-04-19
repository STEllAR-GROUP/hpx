//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/runtime/actions/action_support.hpp>

#include <hpx/plugins/plugin_registry.hpp>
#include <hpx/plugins/binary_filter_factory.hpp>
#include <hpx/plugins/binary_filter/bzip2_serialization_filter.hpp>

#include <boost/format.hpp>

///////////////////////////////////////////////////////////////////////////////
HPX_REGISTER_PLUGIN_MODULE();
HPX_REGISTER_BINARY_FILTER_FACTORY(
    hpx::plugins::compression::bzip2_serialization_filter,
    bzip2_serialization_filter);

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace plugins { namespace compression
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
            using namespace boost::iostreams::bzip2;

            if (!this->ready())
                init();
            if (eof_)
                return false;

            this->before(src_begin, src_end, dest_begin, dest_end);
            int result = this->compress(flush ? finish : run);
            this->after(src_begin, dest_begin);
            (bzip2_error::check)(result);
            return !(eof_ = result == stream_end);
        }

        bool bzip2_compdecomp::load(char const*& src_begin, char const* src_end,
            char*& dest_begin, char* dest_end)
        {
            using namespace boost::iostreams;
            using namespace boost::iostreams::bzip2;

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
            if(result == ok)
                result = this->check_end(src_begin, dest_begin);
            this->after(src_begin, dest_begin);
            (bzip2_error::check)(result);
            eof_ = result == stream_end;
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

    void bzip2_serialization_filter::set_max_length(std::size_t size)
    {
        buffer_.reserve(size);
    }

    ///////////////////////////////////////////////////////////////////////////
    std::size_t bzip2_serialization_filter::load_impl(void* dst,
        std::size_t dst_count, void const* src, std::size_t src_count)
    {
        char const* src_begin = static_cast<char const*>(src);
        char* dst_begin = static_cast<char*>(dst);
        compdecomp_.load(src_begin, src_begin+src_count, dst_begin,
            dst_begin+dst_count);
        return src_begin-static_cast<char const*>(src);
    }

    std::size_t bzip2_serialization_filter::init_data(
        char const* buffer, std::size_t size, std::size_t buffer_size)
    {
        buffer_.resize(buffer_size);
        std::size_t s = load_impl(buffer_.data(), buffer_size, buffer, size);
        if (s > size)
        {
            HPX_THROW_EXCEPTION(serialization_error,
                "bzip2_serialization_filter::load",
                boost::str(boost::format("decompression failure, number of "
                    "bytes expected: %d, number of bytes decoded: %d") %
                        size % s) );
            return 0;
        }
        current_ = 0;
        return buffer_.size();
    }

    ///////////////////////////////////////////////////////////////////////////
    void bzip2_serialization_filter::load(void* dst, std::size_t dst_count)
    {
        if (current_+dst_count > buffer_.size())
        {
            HPX_THROW_EXCEPTION(serialization_error,
                    "bzip2_serialization_filter::load",
                    "archive data bstream is too short");
            return;
        }

        std::memcpy(dst, &buffer_[current_], dst_count);
        current_ += dst_count;
    }

    ///////////////////////////////////////////////////////////////////////////
    void bzip2_serialization_filter::save(void const* src, std::size_t src_count)
    {
        char const* src_begin = static_cast<char const*>(src);
        std::copy(src_begin, src_begin+src_count, std::back_inserter(buffer_));
    }

    ///////////////////////////////////////////////////////////////////////////
    bool bzip2_serialization_filter::flush(void* dst, std::size_t dst_count,
        std::size_t& written)
    {
        // compress everything in one go
        char* dst_begin = static_cast<char*>(dst);
        char const* src_begin = buffer_.data();
        bool eof = compdecomp_.save(src_begin, src_begin+buffer_.size(),
                dst_begin, dst_begin+dst_count, true);
        written = dst_begin-static_cast<char*>(dst);
        return !eof;
    }
}}}

