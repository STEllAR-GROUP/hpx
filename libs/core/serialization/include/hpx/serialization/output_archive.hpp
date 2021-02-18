//  Copyright (c) 2014 Thomas Heller
//  Copyright (c) 2015 Anton Bikineev
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/config/endian.hpp>
#include <hpx/assert.hpp>
#include <hpx/serialization/basic_archive.hpp>
#include <hpx/serialization/detail/polymorphic_nonintrusive_factory.hpp>
#include <hpx/serialization/detail/raw_ptr.hpp>
#include <hpx/serialization/output_container.hpp>
#include <hpx/serialization/traits/is_bitwise_serializable.hpp>

#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace serialization {

    namespace detail {

        template <typename Container>
        inline std::unique_ptr<erased_output_container> create_output_container(
            Container& buffer, std::vector<serialization_chunk>* chunks,
            binary_filter* filter, std::false_type)
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
                        buffer, chunks));
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
                        vector_chunker>(buffer, chunks));
                }
            }
            return res;
        }

        template <typename Container>
        inline std::unique_ptr<erased_output_container> create_output_container(
            Container& buffer, std::vector<serialization_chunk>* chunks,
            binary_filter* filter, std::true_type)
        {
            std::unique_ptr<erased_output_container> res;
            if (filter == nullptr)
            {
                res.reset(new output_container<Container, counting_chunker>(
                    buffer, chunks));
            }
            else
            {
                res.reset(
                    new filtered_output_container<Container, counting_chunker>(
                        buffer, chunks));
            }
            return res;
        }
    }    // namespace detail

    ////////////////////////////////////////////////////////////////////////////
    struct output_archive : basic_archive<output_archive>
    {
    private:
        static std::uint32_t make_flags(
            std::uint32_t flags, std::vector<serialization_chunk>* chunks)
        {
            return flags |
                (chunks == nullptr ? archive_flags::disable_data_chunking :
                                     archive_flags::no_archive_flags);
        }

    public:
        using base_type = basic_archive<output_archive>;

        template <typename Container>
        output_archive(Container& buffer, std::uint32_t flags = 0U,
            std::vector<serialization_chunk>* chunks = nullptr,
            binary_filter* filter = nullptr)
          : base_type(make_flags(flags, chunks))
          , buffer_(detail::create_output_container(buffer, chunks, filter,
                typename traits::serialization_access_data<
                    Container>::preprocessing_only()))
        {
            // endianness needs to be saved separately as it is needed to
            // properly interpret the flags

            // FIXME: make bool once integer compression is implemented
            std::uint64_t endianness =
                this->base_type::endian_big() ? ~0ul : 0ul;
            save(endianness);

            // send flags sent by the other end to make sure both ends have
            // the same assumptions about the archive format
            save(this->flags_);

            bool has_filter = filter != nullptr;
            save(has_filter);

            if (has_filter && enable_compression())
            {
                *this << detail::raw_ptr(filter);
                buffer_->set_filter(filter);
            }
        }

        std::size_t bytes_written() const
        {
            return size_;
        }

        std::size_t get_num_chunks() const
        {
            return buffer_->get_num_chunks();
        }

        // this function is needed to avoid a MSVC linker error
        std::size_t current_pos() const
        {
            return basic_archive<output_archive>::current_pos();
        }

        void reset()
        {
            buffer_->reset();
            basic_archive<output_archive>::reset();
        }

        void flush()
        {
            buffer_->flush();
        }

        bool is_preprocessing() const
        {
            return buffer_->is_preprocessing();
        }

    protected:
        friend struct basic_archive<output_archive>;

        template <class T>
        friend class array;

        template <typename T>
        void invoke_impl(T const& t)
        {
            save(t);
        }

        template <typename T>
        typename std::enable_if<!std::is_integral<T>::value &&
            !std::is_enum<T>::value>::type
        save(T const& t)
        {
            using use_optimized = hpx::traits::is_bitwise_serializable<T>;

            save_bitwise(t, use_optimized());
        }

        template <typename T>
        typename std::enable_if<std::is_integral<T>::value ||
            std::is_enum<T>::value>::type
        save(T t)    //-V659
        {
            save_integral(t, std::is_unsigned<T>());
        }

        void save(float f)
        {
            save_binary(&f, sizeof(float));
        }

        void save(double d)
        {
            save_binary(&d, sizeof(double));
        }

        void save(char c)
        {
            save_binary(&c, sizeof(char));
        }

        void save(bool b)
        {
            HPX_ASSERT(0 == static_cast<int>(b) || 1 == static_cast<int>(b));
            save_binary(&b, sizeof(bool));
        }

        template <typename T>
        void save_bitwise(T const& t, std::false_type)
        {
            save_nonintrusively_polymorphic(
                t, hpx::traits::is_nonintrusive_polymorphic<T>());
        }

        // FIXME: think about removing this commented stuff below
        // and adding new free function save_bitwise
        template <typename T>
        void save_bitwise(T const& t, std::true_type)
        {
            static_assert(!std::is_abstract<T>::value,
                "Can not bitwise serialize a class that is abstract");
            if (disable_array_optimization())
            {
                access::serialize(*this, t, 0);
            }
            else
            {
                save_binary(&t, sizeof(t));
            }
        }

        template <typename T>
        void save_nonintrusively_polymorphic(T const& t, std::false_type)
        {
            access::serialize(*this, t, 0);
        }

        template <typename T>
        void save_nonintrusively_polymorphic(T const& t, std::true_type)
        {
            detail::polymorphic_nonintrusive_factory::instance().save(*this, t);
        }

        template <typename T>
        void save_integral(T val, std::false_type)
        {
            save_integral_impl(static_cast<std::int64_t>(val));
        }

        template <typename T>
        void save_integral(T val, std::true_type)
        {
            save_integral_impl(static_cast<std::uint64_t>(val));
        }

        template <class Promoted>
        void save_integral_impl(Promoted l)
        {
            const std::size_t size = sizeof(Promoted);
            char* cptr = reinterpret_cast<char*>(&l);    //-V206

            const bool endianess_differs =
                endian::native == endian::big ? endian_little() : endian_big();
            if (endianess_differs)
                reverse_bytes(size, cptr);

            save_binary(cptr, size);
        }

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

        std::unique_ptr<erased_output_container> buffer_;
    };
}}    // namespace hpx::serialization

#include <hpx/config/warnings_suffix.hpp>
