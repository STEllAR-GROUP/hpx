//  Copyright (c) 2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_SERIALIZATION_ARCHIVE_HPP
#define HPX_SERIALIZATION_ARCHIVE_HPP

#include <hpx/config.hpp>
#include <hpx/serialization/container.hpp>

#include <hpx/traits/is_bitwise_serializable.hpp>

#include <boost/static_assert.hpp>
#include <boost/type_traits/is_pointer.hpp>
#include <boost/type_traits/is_arithmetic.hpp>
#include <boost/type_traits/is_unsigned.hpp>
#include <boost/type_traits/is_abstract.hpp>

#include <algorithm>
#include <map>

#include <typeinfo>
#include <iostream>

namespace hpx { namespace serialization {
    namespace detail
    {
        struct ptr_helper
        {
            virtual ~ptr_helper() {}
        };
    }

    enum archive_flags
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

    struct output_archive;
    struct input_archive;

    template <typename Helper>
    Helper & tracked_pointer(input_archive & ar, std::size_t pos);

    template <typename Archive>
    struct archive
    {
        static const std::size_t npos = -1;

        archive(boost::uint32_t flags, HPX_STD_UNIQUE_PTR<container> buffer)
          : flags_(flags)
          , buffer_(std::move(buffer))
          , size_(0)
        {}

        virtual ~archive()
        {}

        template <typename T>
        void invoke(T * t)
        {
            BOOST_STATIC_ASSERT_MSG(!boost::is_pointer<T *>::value,
                "HPX does not support serialization of raw pointers. "
                "Please use smart pointers.");
        }

        template <typename T>
        void invoke(T & t)
        {
            static_cast<Archive *>(this)->invoke_impl(t);
        }

        bool enable_compression() const
        {
            return flags_ & hpx::serialization::enable_compression;
        }

        bool endian_big() const
        {
            return flags_ & hpx::serialization::endian_big;
        }

        bool endian_little() const
        {
            return flags_ & hpx::serialization::endian_little;
        }

        bool disable_array_optimization() const
        {
            return flags_ & hpx::serialization::disable_array_optimization;
        }

        bool disable_data_chunking() const
        {
            return flags_ & hpx::serialization::disable_data_chunking;
        }

        boost::uint64_t flags_;
        HPX_STD_UNIQUE_PTR<container> buffer_;
        std::size_t size_;
    };

    template <typename Archive>
    std::size_t & current_pos(Archive & ar)
    {
        return ar.size_;
    }

    template <typename Archive>
    bool has_array_optimization(Archive & ar)
    {
        return !ar.disable_array_optimization();
    }

    template <typename Archive>
    void save_binary(Archive & ar, void const * address, std::size_t count)
    {
        return ar.save_binary(address, count);
    }
    template <typename Archive>
    void load_binary(Archive & ar, void * address, std::size_t count)
    {
        return ar.load_binary(address, count);
    }
}}

#endif
