//  Copyright (c) 2014 Thomas Heller
//  Copyright (c) 2015 Anton Bikineev
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/config/endian.hpp>
#include <hpx/serialization/config/defines.hpp>
#include <hpx/assert.hpp>
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
#include <utility>
#include <vector>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace serialization {

    struct input_archive : basic_archive<input_archive>
    {
        using base_type = basic_archive<input_archive>;

        template <typename Container>
        input_archive(Container& buffer, std::size_t inbound_data_size = 0,
            const std::vector<serialization_chunk>* chunks = nullptr)
          : base_type(0U)
          , buffer_(new input_container<Container>(
                buffer, chunks, inbound_data_size))
        {
            // endianness needs to be saves separately as it is needed to
            // properly interpret the flags

            // FIXME: make bool once integer compression is implemented
            std::uint64_t endianness = 0ul;
            load(endianness);
            if (endianness)
                this->base_type::flags_ = hpx::serialization::endian_big;

            // load flags sent by the other end to make sure both ends have
            // the same assumptions about the archive format
            std::uint32_t flags = 0;
            load(flags);
            this->base_type::flags_ = flags;

            bool has_filter = false;
            load(has_filter);

            serialization::binary_filter* filter = nullptr;
            if (has_filter && enable_compression())
            {
                *this >> detail::raw_ptr(filter);
                buffer_->set_filter(filter);
            }
        }

        template <typename T>
        void invoke_impl(T& t)
        {
            load(t);
        }

        template <typename T>
        std::enable_if_t<!std::is_integral_v<T> && !std::is_enum_v<T>> load(
            T& t)
        {
            using use_optimized = std::integral_constant<bool,
                hpx::traits::is_bitwise_serializable_v<T> ||
                    !hpx::traits::is_not_bitwise_serializable_v<T>>;

            load_bitwise(t, use_optimized());
        }

        template <typename T>
        std::enable_if_t<std::is_integral_v<T> || std::is_enum_v<T>> load(
            T& t)    //-V659
        {
            load_integral(t, std::is_unsigned<T>());
        }

        void load(float& f)
        {
            load_binary(&f, sizeof(float));
        }

        void load(double& d)
        {
            load_binary(&d, sizeof(double));
        }

        void load(char& c)
        {
            load_binary(&c, sizeof(char));
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

        std::size_t bytes_read() const
        {
            return size_;
        }

        // this function is needed to avoid a MSVC linker error
        std::size_t current_pos() const
        {
            return basic_archive<input_archive>::current_pos();
        }

    private:
        friend struct basic_archive<input_archive>;

        template <typename T>
        friend class array;

        template <typename T>
        void load_bitwise(T& t, std::false_type)
        {
            load_nonintrusively_polymorphic(
                t, hpx::traits::is_nonintrusive_polymorphic<T>());
        }

        template <typename T>
        void load_bitwise(T& t, std::true_type)
        {
            static_assert(!std::is_abstract<T>::value,
                "Can not bitwise serialize a class that is abstract");

            bool archive_endianess_differs =
                endian::native == endian::big ? endian_little() : endian_big();

#if !defined(HPX_SERIALIZATION_HAVE_ALL_TYPES_ARE_BITWISE_SERIALIZABLE)
            if (disable_array_optimization() || archive_endianess_differs)
            {
                access::serialize(*this, t, 0);
            }
            else
            {
                load_binary(&t, sizeof(t));
            }
#else
            (void) archive_endianess_differs;
            HPX_ASSERT(
                !(disable_array_optimization() || archive_endianess_differs));
            load_binary(&t, sizeof(t));
#endif
        }

        template <class T>
        void load_nonintrusively_polymorphic(T& t, std::false_type)
        {
            access::serialize(*this, t, 0);
        }

        template <class T>
        void load_nonintrusively_polymorphic(T& t, std::true_type)
        {
            detail::polymorphic_nonintrusive_factory::instance().load(*this, t);
        }

        template <typename T>
        void load_integral(T& val, std::false_type)
        {
            std::int64_t l;
            load_integral_impl(l);
            val = static_cast<T>(l);
        }

        template <typename T>
        void load_integral(T& val, std::true_type)
        {
            std::uint64_t ul;
            load_integral_impl(ul);
            val = static_cast<T>(ul);
        }
        template <class Promoted>
        void load_integral_impl(Promoted& l)
        {
            const std::size_t size = sizeof(Promoted);
            char* cptr = reinterpret_cast<char*>(&l);    //-V206
            load_binary(cptr, static_cast<std::size_t>(size));

            const bool endianess_differs =
                endian::native == endian::big ? endian_little() : endian_big();
            if (endianess_differs)
                reverse_bytes(size, cptr);
        }

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

        std::unique_ptr<erased_input_container> buffer_;
    };

    //
}}    // namespace hpx::serialization

#include <hpx/config/warnings_suffix.hpp>
