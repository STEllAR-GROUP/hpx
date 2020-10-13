//  Copyright (c) 2014 Thomas Heller
//  Copyright (c) 2015 Anton Bikineev
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/modules/datastructures.hpp>
#include <hpx/serialization/detail/extra_archive_data.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <map>
#include <type_traits>
#include <utility>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace serialization {

    namespace detail {

        struct ptr_helper
        {
            virtual ~ptr_helper() = default;
        };
    }    // namespace detail

    enum archive_flags
    {
        no_archive_flags = 0x00000000,
        enable_compression = 0x00002000,
        endian_big = 0x00004000,
        endian_little = 0x00008000,
        disable_array_optimization = 0x00010000,
        disable_data_chunking = 0x00020000,
        all_archive_flags = 0x0003e000    // all of the above
    };

    void HPX_FORCEINLINE reverse_bytes(std::size_t size, char* address)
    {
        std::reverse(address, address + size);
    }

    template <typename Archive>
    struct basic_archive
    {
        static const std::uint64_t npos = std::uint64_t(-1);

    protected:
        basic_archive(std::uint32_t flags)
          : flags_(flags)
          , size_(0)
        {
        }

        basic_archive(basic_archive const&) = delete;
        basic_archive& operator=(basic_archive const&) = delete;

    public:
        virtual ~basic_archive() = default;

        template <typename T>
        void invoke(T& t)
        {
            static_assert(!std::is_pointer<T>::value,
                "HPX does not support serialization of raw pointers. "
                "Please use smart pointers.");

            static_cast<Archive*>(this)->invoke_impl(t);
        }

        bool enable_compression() const
        {
            return (flags_ & hpx::serialization::enable_compression) ? true :
                                                                       false;
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
                true :
                false;
        }

        bool disable_data_chunking() const
        {
            return (flags_ & hpx::serialization::disable_data_chunking) ? true :
                                                                          false;
        }

        std::uint32_t flags() const
        {
            return flags_;
        }

        // Archives can be used to do 'fake' serialization, in which case no
        // data is being stored/restored and no side effects should be
        // performed during serialization/de-serialization.
        bool is_preprocessing() const
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

        void reset()
        {
            size_ = 0;
            extra_data_.reset();
        }

        // access extra data stored
        template <typename T>
        T& get_extra_data()
        {
            return extra_data_.get<T>();
        }

        // try accessing extra data stored, might return nullptr
        template <typename T>
        T* try_get_extra_data()
        {
            return extra_data_.try_get<T>();
        }

    protected:
        std::uint32_t flags_;
        std::size_t size_;
        detail::extra_archive_data extra_data_;
    };

    template <typename Archive>
    inline void save_binary(Archive& ar, void const* address, std::size_t count)
    {
        return ar.basic_archive<Archive>::save_binary(address, count);
    }

    template <typename Archive>
    inline void load_binary(Archive& ar, void* address, std::size_t count)
    {
        return ar.basic_archive<Archive>::load_binary(address, count);
    }

    template <typename Archive>
    inline std::size_t current_pos(const Archive& ar)
    {
        return ar.current_pos();
    }
}}    // namespace hpx::serialization

#include <hpx/config/warnings_suffix.hpp>
