//  Copyright (c) 2014 Thomas Heller
//  Copyright (c) 2015 Anton Bikineev
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_SERIALIZATION_BASIC_ARCHIVE_HPP
#define HPX_SERIALIZATION_BASIC_ARCHIVE_HPP

#include <hpx/config.hpp>
#include <hpx/traits/is_bitwise_serializable.hpp>

#include <boost/static_assert.hpp>
#include <boost/cstdint.hpp>
#include <boost/type_traits/is_pointer.hpp>

#include <algorithm>
#include <map>
#include <iostream>

namespace hpx { namespace serialization
{
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

    void BOOST_FORCEINLINE
    reverse_bytes(char size, char* address)
    {
        std::reverse(address, address + size);
    }

    template <typename Archive>
    struct HPX_EXPORT basic_archive
    {
        static const boost::uint64_t npos = -1;

        basic_archive(boost::uint32_t flags)
          : flags_(flags)
          , size_(0)
        {}

        virtual ~basic_archive()
        {}

        template <typename T>
        void invoke(T & t)
        {
            BOOST_STATIC_ASSERT_MSG(!boost::is_pointer<T>::value,
                "HPX does not support serialization of raw pointers. "
                "Please use smart pointers.");

            static_cast<Archive *>(this)->invoke_impl(t);
        }

        bool enable_compression() const
        {
            return (flags_ & hpx::serialization::enable_compression) ?
                true : false;
        }

        bool endian_big() const
        {
            return (flags_ & hpx::serialization::endian_big) ? true : false;
        }

        bool endian_little() const
        {
            return (flags_ & hpx::serialization::endian_little) ? true : false;
        }

        bool disable_array_optimization() const
        {
            return (flags_ & hpx::serialization::disable_array_optimization) ?
                true : false;
        }

        bool disable_data_chunking() const
        {
            return (flags_ & hpx::serialization::disable_data_chunking) ?
                true : false;
        }

        boost::uint32_t flags() const
        {
            return flags_;
        }

        // Archives can be used to do 'fake' serialization, in which case no
        // data is being stored/restored and no side effects should be
        // performed during serialization/de-serialization.
        bool is_saving() const
        {
            return false;
        }

        std::size_t current_pos() const
        {
            return size_;
        }

        void save_binary(void const* address, std::size_t count)
        {
            static_cast<Archive*>(this)->save_binary(address, count);
        }

        void load_binary(void* address, std::size_t count)
        {
            static_cast<Archive*>(this)->load_binary(address, count);
        }

    protected:
        boost::uint32_t flags_;
        std::size_t size_;
    };

    template <typename Archive>
    inline
    void save_binary(Archive & ar, void const * address, std::size_t count)
    {
        return ar.basic_archive<Archive>::save_binary(address, count);
    }

    template <typename Archive>
    inline
    void load_binary(Archive & ar, void * address, std::size_t count)
    {
        return ar.basic_archive<Archive>::load_binary(address, count);
    }

    template <typename Archive>
    inline
    std::size_t current_pos(const Archive& ar)
    {
        return ar.current_pos();
    }
}}

#endif
