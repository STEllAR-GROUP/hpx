//  Copyright (c) 2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/serialization/input_archive.hpp>

namespace hpx { namespace serialization {

    void register_pointer(input_archive & ar, std::size_t pos, HPX_STD_UNIQUE_PTR<detail::ptr_helper> helper)
    {
        ar.register_pointer(pos, std::move(helper));
    }

    void input_archive::load_impl(boost::int64_t & l)
    {
        const std::size_t size = sizeof(boost::int64_t);
        char* cptr = reinterpret_cast<char *>(&l);
        load_binary(cptr, static_cast<std::size_t>(size));

#ifdef BOOST_BIG_ENDIAN
        if (endian_little())
            reverse_bytes(size, cptr);
#else
        if (endian_big())
            reverse_bytes(size, cptr);
#endif
    }

    void input_archive::load_impl(boost::uint64_t & ul)
    {
        const std::size_t size = sizeof(boost::uint64_t);
        char* cptr = reinterpret_cast<char *>(&ul);
        load_binary(cptr, static_cast<std::size_t>(size));

#ifdef BOOST_BIG_ENDIAN
        if (endian_little())
            reverse_bytes(size, cptr);
#else
        if (endian_big())
            reverse_bytes(size, cptr);
#endif
    }

    void input_archive::load_binary(void * address, std::size_t count)
    {
        if (0 == count) return;

        if(disable_data_chunking())
          buffer_->load_binary(address, count);
        else
          buffer_->load_binary_chunk(address, count);

        size_ += count;
    }

}}
