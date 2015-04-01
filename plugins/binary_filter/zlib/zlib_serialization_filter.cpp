//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/actions/action_support.hpp>

#include <hpx/plugins/plugin_registry.hpp>
#include <hpx/plugins/binary_filter_factory.hpp>
#include <hpx/plugins/binary_filter/zlib_serialization_filter.hpp>

#include <boost/format.hpp>

///////////////////////////////////////////////////////////////////////////////
HPX_REGISTER_PLUGIN_MODULE();
HPX_REGISTER_BINARY_FILTER_FACTORY(
    hpx::plugins::compression::zlib_serialization_filter,
    zlib_serialization_filter);

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace plugins { namespace compression
{
    namespace detail
    {
        zlib_compdecomp::zlib_compdecomp(bool compress,
                boost::iostreams::zlib_params const& params)
          : compress_(compress), eof_(false)
        {
            this->init(params, compress_, static_cast<allocator_type&>(*this));
        }

        zlib_compdecomp::~zlib_compdecomp()
        {
            this->reset(compress_, false);
        }

        bool zlib_compdecomp::save(char const*& src_begin, char const* src_end,
            char*& dest_begin, char* dest_end, bool flush)
        {
            using namespace boost::iostreams;
            using namespace boost::iostreams::zlib;

            this->before(src_begin, src_end, dest_begin, dest_end);
            int result = this->xdeflate(flush ? finish : no_flush);
            this->after(src_begin, dest_begin, true);
            (zlib_error::check)(result);
            return !(eof_ = result == stream_end);
        }

        bool zlib_compdecomp::load(char const*& src_begin, char const* src_end,
            char*& dest_begin, char* dest_end)
        {
            using namespace boost::iostreams;
            using namespace boost::iostreams::zlib;

            this->before(src_begin, src_end, dest_begin, dest_end);
            int result = this->xinflate(sync_flush);
            this->after(src_begin, dest_begin, false);
            (zlib_error::check)(result);
            return !(eof_ = result == stream_end);
        }

        void zlib_compdecomp::close()
        {
            eof_ = false;
            this->reset(compress_, true);
        }
    }

    void zlib_serialization_filter::set_max_length(std::size_t size)
    {
        buffer_.reserve(size);
    }

    ///////////////////////////////////////////////////////////////////////////
    std::size_t zlib_serialization_filter::load_impl(void* dst,
        std::size_t dst_count, void const* src, std::size_t src_count)
    {
        char const* src_begin = static_cast<char const*>(src);
        char* dst_begin = static_cast<char*>(dst);
        compdecomp_.load(src_begin, src_begin+src_count, dst_begin,
            dst_begin+dst_count);
        return src_begin-static_cast<char const*>(src);
    }

    std::size_t zlib_serialization_filter::init_data(
        char const* buffer, std::size_t size, std::size_t buffer_size)
    {
        buffer_.resize(buffer_size);
        std::size_t s = load_impl(buffer_.data(), buffer_size, buffer, size);
        if (s > size)
        {
            HPX_THROW_EXCEPTION(serialization_error,
                "zlib_serialization_filter::load",
                boost::str(boost::format("decompression failure, number of "
                    "bytes expected: %d, number of bytes decoded: %d") %
                        size % s) );
            return 0;
        }

        current_ = 0;
        return buffer_.size();
    }

    ///////////////////////////////////////////////////////////////////////////
    void zlib_serialization_filter::load(void* dst, std::size_t dst_count)
    {
        if (current_+dst_count > buffer_.size())
        {
            HPX_THROW_EXCEPTION(serialization_error,
                    "zlib_serialization_filter::load",
                    "archive data bstream is too short");
            return;
        }

        std::memcpy(dst, &buffer_[current_], dst_count);
        current_ += dst_count;
    }

    ///////////////////////////////////////////////////////////////////////////
    void zlib_serialization_filter::save(void const* src,
        std::size_t src_count)
    {
        char const* src_begin = static_cast<char const*>(src);
        std::copy(src_begin, src_begin+src_count, std::back_inserter(buffer_));
    }

    ///////////////////////////////////////////////////////////////////////////
    bool zlib_serialization_filter::flush(void* dst,
        std::size_t dst_count, std::size_t& written)
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

