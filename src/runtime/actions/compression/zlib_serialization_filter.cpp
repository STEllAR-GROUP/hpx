//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>

#if defined(HPX_HAVE_ZLIB_COMPRESSION)
#include <hpx/runtime/actions/action_support.hpp>
#include <hpx/runtime/actions/compression/zlib_serialization_filter.hpp>
#include <hpx/runtime/actions/guid_initialization.hpp>
#include <hpx/util/void_cast.hpp>

#include <boost/format.hpp>

///////////////////////////////////////////////////////////////////////////////
HPX_SERIALIZATION_REGISTER_TYPE_DEFINITION(hpx::actions::zlib_serialization_filter);
HPX_REGISTER_BASE_HELPER(hpx::actions::zlib_serialization_filter,
    zlib_serialization_filter);

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace actions
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

            this->before(src_begin, src_end, dest_begin, dest_end);
            int result = this->xdeflate(flush ? zlib::finish : zlib::no_flush);
            this->after(src_begin, dest_begin, true);
            (zlib_error::check)(result);
            return !(eof_ = (result == zlib::stream_end ? true : false));
        }

        bool zlib_compdecomp::load(char const*& src_begin, char const* src_end,
            char*& dest_begin, char* dest_end)
        {
            using namespace boost::iostreams;

            this->before(src_begin, src_end, dest_begin, dest_end);
            int result = this->xinflate(zlib::sync_flush);
            this->after(src_begin, dest_begin, false);
            (zlib_error::check)(result);
            return !(eof_ = (result == zlib::stream_end ? true : false));
        }

        void zlib_compdecomp::close()
        {
            eof_ = false;
            this->reset(compress_, true);
        }
    }

    zlib_serialization_filter::~zlib_serialization_filter()
    {
        detail::guid_initialization<zlib_serialization_filter>();
    }

    void zlib_serialization_filter::register_base()
    {
        util::void_cast_register_nonvirt<
            zlib_serialization_filter, util::binary_filter>();
    }

    std::size_t zlib_serialization_filter::load(void* dst,
        std::size_t dst_count, void const* src, std::size_t src_count)
    {
        char const* src_begin = static_cast<char const*>(src);
        char* dst_begin = static_cast<char*>(dst);
        compdecomp_.load(src_begin, src_begin+src_count, dst_begin,
            dst_begin+dst_count);

        if (dst_begin-static_cast<char*>(dst) != dst_count)
        {
            HPX_THROW_EXCEPTION(serialization_error,
                "zlib_serialization_filter::load",
                boost::str(boost::format("decompression failure, number of "
                    "bytes expected: %d, number of bytes decoded: %d") %
                        dst_count % (dst_begin-static_cast<char*>(dst)) ));
            return 0;
        }
        return src_begin-static_cast<char const*>(src);
    }

    std::size_t zlib_serialization_filter::save(void* dst,
        std::size_t dst_count, void const* src, std::size_t src_count)
    {
        char const* src_begin = static_cast<char const*>(src);
        char* dst_begin = static_cast<char*>(dst);
        compdecomp_.save(src_begin, src_begin+src_count, dst_begin,
            dst_begin+dst_count);

        return dst_begin-static_cast<char*>(dst);
    }

    std::size_t zlib_serialization_filter::flush(void* dst,
        std::size_t dst_count)
    {
        char dummy = '\0';
        char const* src_begin = &dummy;
        char* dst_begin = static_cast<char*>(dst);
        if (compdecomp_.save(src_begin, src_begin, dst_begin,
                dst_begin+dst_count, true))
        {
            HPX_THROW_EXCEPTION(serialization_error,
                "zlib_serialization_filter::flush",
                "compression failure, flushing did not reach end of data");
            return 0;
        }
        return dst_begin-static_cast<char*>(dst);
    }
}}

#endif
