//  Copyright (c) 2014 Thomas Heller
//  Copyright (c) 2015 Anton Bikineev
//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/config/endian.hpp>
#include <hpx/serialization/config/defines.hpp>
#include <hpx/assert.hpp>
#include <hpx/serialization/detail/extra_archive_data.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
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

    enum class archive_flags
    {
        no_archive_flags = 0x00000000,
        enable_compression = 0x00002000,
        endian_big = 0x00004000,
        endian_little = 0x00008000,
        disable_array_optimization = 0x00010000,
        disable_data_chunking = 0x00020000,
        archive_is_saving = 0x00040000,
        archive_is_preprocessing = 0x00080000,
        all_archive_flags = 0x000fe000    // all of the above
    };

#if defined(HPX_SERIALIZATION_HAVE_SUPPORTS_ENDIANESS)
    HPX_FORCEINLINE void reverse_bytes(std::size_t size, char* address)
    {
        std::reverse(address, address + size);
    }
#endif

    template <typename Archive>
    struct basic_archive
    {
        static constexpr std::uint64_t npos = std::uint64_t(-1);

    protected:
        explicit constexpr basic_archive(std::uint32_t flags) noexcept
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
            static_cast<Archive*>(this)->invoke_impl(t);
        }

        constexpr bool archive_is_saving() const noexcept
        {
            return bool(
                flags_ & std::uint32_t(archive_flags::archive_is_saving));
        }

        constexpr bool enable_compression() const noexcept
        {
            return bool(
                flags_ & std::uint32_t(archive_flags::enable_compression));
        }

        constexpr bool endian_big() const noexcept
        {
            return bool(flags_ & std::uint32_t(archive_flags::endian_big));
        }

        constexpr bool endian_little() const noexcept
        {
            return bool(flags_ & std::uint32_t(archive_flags::endian_little));
        }

#if defined(HPX_SERIALIZATION_HAVE_SUPPORTS_ENDIANESS)
        constexpr bool endianess_differs() const noexcept
        {
            return endian::native == endian::big ? endian_little() :
                                                   endian_big();
        }
#else
        static constexpr bool endianess_differs() noexcept
        {
            return false;
        }
#endif

        constexpr bool disable_array_optimization() const noexcept
        {
            return bool(flags_ &
                std::uint32_t(archive_flags::disable_array_optimization));
        }

        constexpr bool disable_data_chunking() const noexcept
        {
            return bool(
                flags_ & std::uint32_t(archive_flags::disable_data_chunking));
        }

        constexpr std::uint32_t flags() const noexcept
        {
            return flags_;
        }

        // Archives can be used to do 'fake' serialization, in which case no
        // data is being stored/restored and no side effects should be
        // performed during serialization/de-serialization.
        constexpr bool is_preprocessing() const noexcept
        {
            return bool(flags_ &
                std::uint32_t(archive_flags::archive_is_preprocessing));
        }

        constexpr std::size_t current_pos() const noexcept
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
        T* try_get_extra_data() const noexcept
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
        ar.save_binary(address, count);
    }

    template <typename Archive>
    inline void load_binary(Archive& ar, void* address, std::size_t count)
    {
        ar.load_binary(address, count);
    }

    template <typename Archive>
    inline std::size_t current_pos(Archive const& ar) noexcept
    {
        return ar.current_pos();
    }
}}    // namespace hpx::serialization

#include <hpx/config/warnings_suffix.hpp>
