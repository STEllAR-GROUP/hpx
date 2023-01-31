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
#include <hpx/serialization/basic_archive.hpp>
#include <hpx/serialization/detail/polymorphic_nonintrusive_factory.hpp>
#include <hpx/serialization/detail/raw_ptr.hpp>
#include <hpx/serialization/output_container.hpp>
#include <hpx/serialization/traits/is_bitwise_serializable.hpp>
#include <hpx/serialization/traits/is_not_bitwise_serializable.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <vector>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx::serialization {

    namespace detail {

        template <typename Container>
        std::unique_ptr<erased_output_container> create_output_container(
            Container& buffer, std::vector<serialization_chunk>* chunks,
            binary_filter* filter,
            std::size_t zero_copy_serialization_threshold, std::false_type)
        {
            std::unique_ptr<erased_output_container> res;
            if (filter == nullptr)
            {
                if (chunks == nullptr)
                {
                    res.reset(
                        new output_container<Container, basic_chunker>(buffer));
                }
                else
                {
                    res.reset(new output_container<Container, vector_chunker>(
                        buffer, chunks, zero_copy_serialization_threshold));
                }
            }
            else
            {
                if (chunks == nullptr)
                {
                    res.reset(
                        new filtered_output_container<Container, basic_chunker>(
                            buffer));
                }
                else
                {
                    res.reset(new filtered_output_container<Container,
                        vector_chunker>(
                        buffer, chunks, zero_copy_serialization_threshold));
                }
            }
            return res;
        }

        template <typename Container>
        std::unique_ptr<erased_output_container> create_output_container(
            Container& buffer, std::vector<serialization_chunk>* chunks,
            binary_filter* filter,
            std::size_t zero_copy_serialization_threshold, std::true_type)
        {
            std::unique_ptr<erased_output_container> res;
            if (filter == nullptr)
            {
                res.reset(new output_container<Container, counting_chunker>(
                    buffer, chunks, zero_copy_serialization_threshold));
            }
            else
            {
                res.reset(
                    new filtered_output_container<Container, counting_chunker>(
                        buffer, chunks, zero_copy_serialization_threshold));
            }
            return res;
        }
    }    // namespace detail

    ////////////////////////////////////////////////////////////////////////////
    struct output_archive : basic_archive<output_archive>
    {
    private:
        static constexpr std::uint32_t make_flags(std::uint32_t flags,
            std::vector<serialization_chunk> const* chunks) noexcept
        {
            return flags |
                static_cast<std::uint32_t>(archive_flags::archive_is_saving) |
                static_cast<std::uint32_t>(chunks == nullptr ?
                        archive_flags::disable_data_chunking :
                        archive_flags::no_archive_flags);
        }

    public:
        using base_type = basic_archive<output_archive>;

        template <typename Container>
        explicit output_archive(Container& buffer, std::uint32_t flags = 0U,
            std::vector<serialization_chunk>* chunks = nullptr,
            binary_filter* filter = nullptr,
            std::size_t zero_copy_serialization_threshold = 0)
          : base_type(make_flags(flags, chunks))
          , buffer_(detail::create_output_container(buffer, chunks, filter,
                zero_copy_serialization_threshold,
                typename traits::serialization_access_data<
                    Container>::preprocessing_only()))
        {
            // cache the preprocessing flag in the base class to avoid asking
            // the buffer repeatedly
            if (buffer_->is_preprocessing())
            {
                flags_ = flags_ |
                    static_cast<std::uint32_t>(
                        archive_flags::archive_is_preprocessing);
            }

            // endianness needs to be saved separately as it is needed to
            // properly interpret the flags
            //
            // FIXME: make bool once integer compression is implemented
            std::uint64_t const endianness = endian_big() ? ~0ul : 0ul;
            save(endianness);

            // send flags sent by the other end to make sure both ends have
            // the same assumptions about the archive format
            save(flags_);

            // send the zero-copy limit
            save(static_cast<std::uint64_t>(zero_copy_serialization_threshold));

            bool const has_filter = filter != nullptr;
            save(has_filter);

            if (has_filter && enable_compression())
            {
                *this << detail::raw_ptr(filter);
                buffer_->set_filter(filter);
            }
        }

        template <typename Container>
        output_archive(Container& buffer, archive_flags flags,
            std::vector<serialization_chunk>* chunks = nullptr,
            binary_filter* filter = nullptr,
            std::size_t zero_copy_serialization_threshold = 0)
          : output_archive(buffer, static_cast<std::uint32_t>(flags), chunks,
                filter, zero_copy_serialization_threshold)
        {
        }

        [[nodiscard]] constexpr std::size_t bytes_written() const noexcept
        {
            return size_;
        }

        [[nodiscard]] std::size_t get_num_chunks() const noexcept
        {
            return buffer_->get_num_chunks();
        }

        // this function is needed to avoid a MSVC linker error
        [[nodiscard]] constexpr std::size_t current_pos() const noexcept
        {
            return base_type::current_pos();
        }

        void reset()
        {
            buffer_->reset();
            base_type::reset();
        }

        void flush() const
        {
            buffer_->flush();
        }

        template <typename T>
        HPX_FORCEINLINE void invoke(T const& t)
        {
            save(t);
        }

        template <typename T>
        HPX_FORCEINLINE void invoke_impl(T const& t)
        {
            save(t);
        }

        template <typename T>
        void save(T const& t)
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
                    save_binary(&t, sizeof(t));
                }
                else if constexpr (traits::is_nonintrusive_polymorphic_v<T>)
                {
                    // non-bitwise polymorphic serialization
                    detail::polymorphic_nonintrusive_factory::instance().save(
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

                save_integral(static_cast<std::uint64_t>(t));
            }
            else
            {
                static_assert(sizeof(T) <= sizeof(std::int64_t),
                    "integral type is larger than supported");

                save_integral(static_cast<std::int64_t>(t));
            }
#else
            else if constexpr (std::is_unsigned_v<T>)
            {
                static_assert(sizeof(T) <= sizeof(std::uint64_t),
                    "integral type is larger than supported");

                auto const val = static_cast<std::uint64_t>(t);
                save_binary(&val, sizeof(std::uint64_t));
            }
            else
            {
                static_assert(sizeof(T) <= sizeof(std::int64_t),
                    "integral type is larger than supported");

                auto const val = static_cast<std::int64_t>(t);
                save_binary(&val, sizeof(std::int64_t));
            }
#endif
        }

        void save(float f)
        {
            save_binary(&f, sizeof(float));
        }

        void save(double d)
        {
            save_binary(&d, sizeof(double));
        }

        void save(long double d)
        {
            save_binary(&d, sizeof(long double));
        }

        void save(char c)
        {
            save_binary(&c, sizeof(char));
        }

        void save(signed char c)
        {
            save_binary(&c, sizeof(signed char));
        }

        void save(unsigned char c)
        {
            save_binary(&c, sizeof(unsigned char));
        }

        void save(bool b)
        {
            HPX_ASSERT(0 == static_cast<int>(b) || 1 == static_cast<int>(b));
            save_binary(&b, sizeof(bool));
        }

#if defined(HPX_SERIALIZATION_HAVE_ALLOW_RAW_POINTER_SERIALIZATION)
        template <typename T>
        void save(T* p)
        {
            save_binary(&p, sizeof(std::size_t));
        }
#endif

    private:
        friend struct basic_archive<output_archive>;

#if defined(HPX_SERIALIZATION_HAVE_SUPPORTS_ENDIANESS)
        template <typename Promoted>
        void save_integral(Promoted l)
        {
            if (endianess_differs())
            {
                reverse_bytes(sizeof(Promoted), reinterpret_cast<char*>(&l));
            }
            save_binary(&l, sizeof(Promoted));
        }
#endif

    public:
        void save_binary(void const* address, std::size_t count)
        {
            if (count == 0)
                return;

            size_ += count;
            buffer_->save_binary(address, count);
        }

        void save_binary_chunk(void const* address, std::size_t count)
        {
            if (count == 0)
                return;

            if (disable_data_chunking())
            {
                size_ += count;
                buffer_->save_binary(address, count);
            }
            else
            {
                // the size might grow if optimizations are not used
                size_ += buffer_->save_binary_chunk(address, count);
            }
        }

    private:
        std::unique_ptr<erased_output_container> buffer_;
    };
}    // namespace hpx::serialization

#include <hpx/config/warnings_suffix.hpp>
