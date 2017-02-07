//  Copyright (c) 2014 Thomas Heller
//  Copyright (c) 2015 Anton Bikineev
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// hpxinspect:nodeprecatedinclude:boost/cstdint.hpp

#ifndef HPX_SERIALIZATION_INPUT_ARCHIVE_HPP
#define HPX_SERIALIZATION_INPUT_ARCHIVE_HPP

#include <hpx/config.hpp>
#include <hpx/runtime/serialization/basic_archive.hpp>
#include <hpx/runtime/serialization/detail/polymorphic_nonintrusive_factory.hpp>
#include <hpx/runtime/serialization/detail/raw_ptr.hpp>
#include <hpx/runtime/serialization/input_container.hpp>
#include <hpx/traits/is_bitwise_serializable.hpp>

#include <boost/cstdint.hpp>

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace serialization
{
    struct HPX_EXPORT input_archive
      : basic_archive<input_archive>
    {
        typedef basic_archive<input_archive> base_type;

        typedef
            std::map<std::uint64_t, detail::ptr_helper_ptr>
            pointer_tracker;

        template <typename Container>
        input_archive(Container & buffer,
            std::size_t inbound_data_size = 0,
            const std::vector<serialization_chunk>* chunks = nullptr)
          : base_type(0U)
          , buffer_(new input_container<Container>(buffer, chunks, inbound_data_size))
        {
            // endianness needs to be saves separately as it is needed to
            // properly interpret the flags

            // FIXME: make bool once integer compression is implemented
            std::uint64_t endianess = 0ul;
            load(endianess);
            if (endianess)
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
        void invoke_impl(T & t)
        {
            load(t);
        }

        template <typename T>
        typename std::enable_if<
            !std::is_integral<T>::value && !std::is_enum<T>::value
        >::type
        load(T & t)
        {
            typedef hpx::traits::is_bitwise_serializable<T> use_optimized;

            load_bitwise(t, use_optimized());
        }

        template <typename T>
        typename std::enable_if<
            std::is_integral<T>::value || std::is_enum<T>::value
        >::type
        load(T & t) //-V659
        {
            load_integral(t, std::is_unsigned<T>());
        }

        void load(float & f)
        {
            load_binary(&f, sizeof(float));
        }

        void load(double & d)
        {
            load_binary(&d, sizeof(double));
        }

        void load(char & c)
        {
            load_binary(&c, sizeof(char));
        }

        void load(bool & b)
        {
            load_binary(&b, sizeof(bool));
            HPX_ASSERT(0 == static_cast<int>(b) || 1 == static_cast<int>(b));
        }

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
        template <class T>
        friend class array;

        template <typename T>
        void load_bitwise(T & t, std::false_type)
        {
            load_nonintrusively_polymorphic(t,
                hpx::traits::is_nonintrusive_polymorphic<T>());
        }

        template <typename T>
        void load_bitwise(T & t, std::true_type)
        {
            static_assert(!std::is_abstract<T>::value,
                "Can not bitwise serialize a class that is abstract");
            if(disable_array_optimization())
            {
                access::serialize(*this, t, 0);
            }
            else
            {
                load_binary(&t, sizeof(t));
            }
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
        void load_integral(T & val, std::false_type)
        {
            std::int64_t l;
            load_integral_impl(l);
            val = static_cast<T>(l);
        }

        template <typename T>
        void load_integral(T & val, std::true_type)
        {
            std::uint64_t ul;
            load_integral_impl(ul);
            val = static_cast<T>(ul);
        }

#if defined(BOOST_HAS_INT128) && !defined(__NVCC__) && \
    !defined(__CUDACC__)
        void load_integral(boost::int128_type& t, std::false_type)
        {
            load_integral_impl(t);
        }

        void load_integral(boost::uint128_type& t, std::true_type)
        {
            load_integral_impl(t);
        }

        // On some platforms (gcc) std::is_integral<int128>::value
        // evaluates to false. Thus, these functions re-route the
        // serialization for those types to the proper implementation
        void load_bitwise(boost::int128_type& t, std::false_type)
        {
            load_integral_impl(t);
        }

        void load_bitwise(boost::int128_type& t, std::true_type)
        {
            load_integral_impl(t);
        }

        void load_bitwise(boost::uint128_type& t, std::false_type)
        {
            load_integral_impl(t);
        }

        void load_bitwise(boost::uint128_type& t, std::true_type)
        {
            load_integral_impl(t);
        }
#endif
        template <class Promoted>
        void load_integral_impl(Promoted& l)
        {
            const std::size_t size = sizeof(Promoted);
            char* cptr = reinterpret_cast<char *>(&l); //-V206
            load_binary(cptr, static_cast<std::size_t>(size));

#ifdef BOOST_BIG_ENDIAN
            if (endian_little())
                reverse_bytes(size, cptr);
#else
            if (endian_big())
                reverse_bytes(size, cptr);
#endif
        }

        void load_binary(void * address, std::size_t count)
        {
            if (0 == count) return;

            buffer_->load_binary(address, count);

            size_ += count;
        }

        void load_binary_chunk(void * address, std::size_t count)
        {
            if (0 == count) return;

            if(disable_data_chunking())
                buffer_->load_binary(address, count);
            else
                buffer_->load_binary_chunk(address, count);

            size_ += count;
        }

        // make functions visible through adl
        friend void register_pointer(input_archive& ar,
                std::uint64_t pos, detail::ptr_helper_ptr helper)
        {
            pointer_tracker& tracker = ar.pointer_tracker_;
            HPX_ASSERT(tracker.find(pos) == tracker.end());

            tracker.insert(std::make_pair(pos, std::move(helper)));
        }

        template <typename Helper>
        friend Helper & tracked_pointer(input_archive& ar, std::uint64_t pos)
        {
            // gcc has some lookup problems when using
            // nested type inside friend function
            std::map<std::uint64_t, detail::ptr_helper_ptr>::iterator
                it = ar.pointer_tracker_.find(pos);
            HPX_ASSERT(it != ar.pointer_tracker_.end());

            return static_cast<Helper &>(*it->second);
        }

        std::unique_ptr<erased_input_container> buffer_;
        pointer_tracker pointer_tracker_;
    };
}}

#include <hpx/config/warnings_suffix.hpp>

#endif
