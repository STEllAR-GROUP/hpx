//  Copyright (c) 2014 Thomas Heller
//  Copyright (c) 2015 Anton Bikineev
//  Copyright (c) 2022-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/config/endian.hpp>
#include <hpx/serialization/config/defines.hpp>
#include <hpx/assert.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/serialization/basic_archive.hpp>
#include <hpx/serialization/detail/polymorphic_nonintrusive_factory.hpp>
#include <hpx/serialization/detail/raw_ptr.hpp>
#include <hpx/serialization/input_container.hpp>
#include <hpx/serialization/traits/is_bitwise_serializable.hpp>
#include <hpx/serialization/traits/is_not_bitwise_serializable.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <vector>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx::serialization {

    struct input_archive : basic_archive<input_archive>
    {
        using base_type = basic_archive<input_archive>;

        template <typename Container>
        explicit input_archive(Container& buffer,
            std::size_t inbound_data_size = 0,
            std::vector<serialization_chunk> const* chunks = nullptr)
          : base_type(0U)
          , buffer_(new input_container<Container>(
                buffer, chunks, inbound_data_size))
        {
            // endianness needs to be saved separately as it is needed to
            // properly interpret the flags

            // FIXME: make bool once integer compression is implemented
            std::uint64_t endianness = 0ul;
            load(endianness);
            if (endianness)
            {
                flags_ = static_cast<std::uint32_t>(
                    hpx::serialization::archive_flags::endian_big);
            }

#if !defined(HPX_SERIALIZATION_HAVE_SUPPORTS_ENDIANESS)
            if ((endianness && (endian::native == endian::little)) ||
                (!endianness && (endian::native == endian::big)))
            {
                HPX_THROW_EXCEPTION(hpx::error::bad_request,
                    "hpx::serialization::input_archive::input_archive",
                    "Converting endianness is not supported by the "
                    "serialization library, please reconfigure HPX with "
                    "-DHPX_SERIALIZATION_WITH_SUPPORTS_ENDIANESS=On");
            }
#endif
            // Load flags sent by the other end to make sure both ends have
            // the same assumptions about the archive format. It is safe to
            // overwrite the flags_ now.
            std::uint32_t flags = 0;
            load(flags);
            flags_ = flags;

            // load the zero-copy limit used by the other end
            std::uint64_t zero_copy_serialization_threshold;
            load(zero_copy_serialization_threshold);
            buffer_->set_zero_copy_serialization_threshold(
                zero_copy_serialization_threshold);

            bool has_filter = false;
            load(has_filter);

            if (has_filter && enable_compression())
            {
                serialization::binary_filter* filter = nullptr;
                *this >> detail::raw_ptr(filter);
                buffer_->set_filter(filter);
            }
        }

        template <typename T>
        HPX_FORCEINLINE void invoke(T& t)
        {
            load(t);
        }

        template <typename T>
        HPX_FORCEINLINE void invoke_impl(T& t)
        {
            load(t);
        }

        template <typename T>
        void load(T& t)
        {
#if !defined(HPX_SERIALIZATION_HAVE_ALLOW_RAW_POINTER_SERIALIZATION)
            static_assert(!std::is_pointer_v<T>,
                "HPX does not support serialization of raw pointers. "
                "Please use smart pointers instead.");
#endif
            if constexpr (!std::is_integral_v<T> && !std::is_enum_v<T>)
            {
                if constexpr (hpx::traits::is_bitwise_serializable_v<T> ||
                    !hpx::traits::is_not_bitwise_serializable_v<T>)
                {
                    // bitwise serialization
                    static_assert(!std::is_abstract_v<T>,
                        "Can not bitwise serialize a class that is abstract");

#if !defined(HPX_SERIALIZATION_HAVE_ALL_TYPES_ARE_BITWISE_SERIALIZABLE)
                    if (disable_array_optimization() || endianess_differs())
                    {
                        access::serialize(*this, t, 0);
                        return;
                    }
#else
                    HPX_ASSERT(
                        !(disable_array_optimization() || endianess_differs()));
#endif
                    load_binary(&t, sizeof(t));
                }
                else if constexpr (traits::is_nonintrusive_polymorphic_v<T>)
                {
                    // non-bitwise polymorphic serialization
                    detail::polymorphic_nonintrusive_factory::instance().load(
                        *this, t);
                }
                else
                {
                    // non-bitwise normal serialization
                    access::serialize(*this, t, 0);
                }
            }
#if defined(HPX_SERIALIZATION_HAVE_SUPPORTS_ENDIANESS)
            else if constexpr (std::is_unsigned_v<T>)
            {
                static_assert(sizeof(T) <= sizeof(std::uint64_t),
                    "integral type is larger than supported");

                std::uint64_t ul;
                load_integral(ul);
                t = static_cast<T>(ul);
            }
            else
            {
                static_assert(sizeof(T) <= sizeof(std::int64_t),
                    "integral type is larger than supported");

                std::int64_t l;
                load_integral(l);
                t = static_cast<T>(l);
            }
#else
            else if constexpr (std::is_unsigned_v<T>)
            {
                static_assert(sizeof(T) <= sizeof(std::uint64_t),
                    "integral type is larger than supported");

                std::uint64_t ul;
                load_binary(&ul, sizeof(std::uint64_t));
                t = static_cast<T>(ul);
            }
            else
            {
                static_assert(sizeof(T) <= sizeof(std::int64_t),
                    "integral type is larger than supported");

                std::int64_t ul;
                load_binary(&ul, sizeof(std::int64_t));
                t = static_cast<T>(ul);
            }
#endif
        }

        void load(float& f)
        {
            load_binary(&f, sizeof(float));
        }

        void load(double& d)
        {
            load_binary(&d, sizeof(double));
        }

        void load(long double& d)
        {
            load_binary(&d, sizeof(long double));
        }

        void load(char& c)
        {
            load_binary(&c, sizeof(char));
        }

        void load(signed char& c)
        {
            load_binary(&c, sizeof(signed char));
        }

        void load(unsigned char& c)
        {
            load_binary(&c, sizeof(unsigned char));
        }

        void load(bool& b)
        {
            load_binary(&b, sizeof(bool));
            HPX_ASSERT(0 == static_cast<int>(b) || 1 == static_cast<int>(b));
        }

#if defined(HPX_SERIALIZATION_HAVE_ALLOW_RAW_POINTER_SERIALIZATION)
        template <typename T>
        void load(T*& p)
        {
            load_binary(&p, sizeof(std::size_t));
        }
#endif

        [[nodiscard]] constexpr std::size_t bytes_read() const noexcept
        {
            return current_pos();
        }

        // this function is needed to avoid a MSVC linker error
        [[nodiscard]] constexpr std::size_t current_pos() const noexcept
        {
            return base_type::current_pos();
        }

    private:
        friend struct basic_archive<input_archive>;

#if defined(HPX_SERIALIZATION_HAVE_SUPPORTS_ENDIANESS)
        template <typename Promoted>
        void load_integral(Promoted& l)
        {
            load_binary(&l, sizeof(Promoted));
            if (endianess_differs())
            {
                reverse_bytes(sizeof(Promoted), reinterpret_cast<char*>(&l));
            }
        }
#endif

    public:
        void load_binary(void* address, std::size_t count)
        {
            if (0 == count)
                return;

            buffer_->load_binary(address, count);

            size_ += count;
        }

        void load_binary_chunk(void* address, std::size_t count)
        {
            if (0 == count)
                return;

            if (disable_data_chunking())
                buffer_->load_binary(address, count);
            else
                buffer_->load_binary_chunk(address, count);

            size_ += count;
        }

    private:
        std::unique_ptr<erased_input_container> buffer_;
    };
}    // namespace hpx::serialization

#include <hpx/config/warnings_suffix.hpp>
