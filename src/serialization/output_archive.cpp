//  Copyright (c) 2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/serialization/output_archive.hpp>

#include <boost/detail/endian.hpp>

namespace hpx { namespace serialization {

    std::size_t track_pointer(output_archive & ar, void * pos)
    {
        return ar.track_pointer(pos);
    }

    void output_archive::save_impl(boost::int64_t l)
    {
        const std::size_t size = sizeof(boost::int64_t);
        char* cptr = reinterpret_cast<char *>(&l);
#ifdef BOOST_BIG_ENDIAN
        if(endian_little())
            reverse_bytes(size, cptr);
#else
        if(endian_big())
            reverse_bytes(size, cptr);
#endif

        save_binary(cptr, size);
    }

    void output_archive::save_impl(boost::uint64_t ul)
    {
        const std::size_t size = sizeof(boost::uint64_t);
        char* cptr = reinterpret_cast<char*>(&ul);

#ifdef BOOST_BIG_ENDIAN
        if(endian_little())
            reverse_bytes(size, cptr);
#else
        if(endian_big())
            reverse_bytes(size, cptr);
#endif

        save_binary(cptr, size);
    }

    void output_archive::save_binary(void const * address, std::size_t count)
    {
        size_ += count;
        if(disable_data_chunking())
          buffer_->save_binary(address, count);
        else
          buffer_->save_binary_chunk(address, count);
    }
}}
