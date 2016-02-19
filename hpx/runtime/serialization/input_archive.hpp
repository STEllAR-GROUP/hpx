//  Copyright (c) 2014 Thomas Heller
//  Copyright (c) 2015 Anton Bikineev
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_SERIALIZATION_INPUT_ARCHIVE_HPP
#define HPX_SERIALIZATION_INPUT_ARCHIVE_HPP

#include <hpx/config.hpp>
#include <hpx/runtime/serialization/basic_archive.hpp>
#include <hpx/runtime/serialization/input_container.hpp>
#include <hpx/runtime/serialization/detail/raw_ptr.hpp>
#include <hpx/runtime/serialization/detail/polymorphic_nonintrusive_factory.hpp>

#include <boost/config.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/type_traits/is_integral.hpp>
#include <boost/type_traits/is_unsigned.hpp>
#include <boost/type_traits/is_enum.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/mpl/or.hpp>

#include <memory>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace serialization
{
    struct HPX_EXPORT input_archive
      : basic_archive<input_archive>
    {
        typedef basic_archive<input_archive> base_type;

        typedef
            std::map<boost::uint64_t, detail::ptr_helper_ptr>
            pointer_tracker;

        template <typename Container>
        input_archive(Container & buffer,
            std::size_t inbound_data_size = 0,
            const std::vector<serialization_chunk>* chunks = 0)
          : base_type(0U)
          , buffer_(new input_container<Container>(buffer, chunks, inbound_data_size))
        {
            // endianness needs to be saves separately as it is needed to
            // properly interpret the flags

            // FIXME: make bool once integer compression is implemented
            boost::uint64_t endianess = 0ul;
            load(endianess);
            if (endianess)
                this->base_type::flags_ = hpx::serialization::endian_big;

            // load flags sent by the other end to make sure both ends have
            // the same assumptions about the archive format
            boost::uint32_t flags = 0;
            load(flags);
            this->base_type::flags_ = flags;

            bool has_filter = false;
            load(has_filter);

            serialization::binary_filter* filter = 0;
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
        typename boost::disable_if<
            boost::mpl::or_<
                boost::is_integral<T>
              , boost::is_enum<T>
            >
        >::type
        load(T & t)
        {
            load_bitwise(t,
                typename hpx::traits::is_bitwise_serializable<T>::type());
        }

        template <typename T>
        typename boost::enable_if<
            boost::mpl::or_<
                boost::is_integral<T>
              , boost::is_enum<T>
            >
        >::type
        load(T & t)
        {
            load_integral(t,
                typename boost::is_unsigned<T>::type());
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
        void load_bitwise(T & t, boost::mpl::false_)
        {
            load_nonintrusively_polymorphic(t,
                hpx::traits::is_nonintrusive_polymorphic<T>());
        }

        template <typename T>
        void load_bitwise(T & t, boost::mpl::true_)
        {
            static_assert(!boost::is_abstract<T>::value,
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
        void load_nonintrusively_polymorphic(T& t, boost::mpl::false_)
        {
            access::serialize(*this, t, 0);
        }

        template <class T>
        void load_nonintrusively_polymorphic(T& t, boost::mpl::true_)
        {
            detail::polymorphic_nonintrusive_factory::instance().load(*this, t);
        }

        template <typename T>
        void load_integral(T & val, boost::mpl::false_)
        {
            boost::int64_t l;
            load_integral_impl(l);
            val = static_cast<T>(l);
        }

        template <typename T>
        void load_integral(T & val, boost::mpl::true_)
        {
            boost::uint64_t ul;
            load_integral_impl(ul);
            val = static_cast<T>(ul);
        }

#if defined(BOOST_HAS_INT128)
        void load_integral(boost::int128_type& t, boost::mpl::false_)
        {
            load_integral_impl(t);
        }

        void load_integral(boost::uint128_type& t, boost::mpl::true_)
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
                boost::uint64_t pos, detail::ptr_helper_ptr helper)
        {
            pointer_tracker& tracker = ar.pointer_tracker_;
            HPX_ASSERT(tracker.find(pos) == tracker.end());

            tracker.insert(std::make_pair(pos, std::move(helper)));
        }

        template <typename Helper>
        friend Helper & tracked_pointer(input_archive& ar, boost::uint64_t pos)
        {
            // gcc has some lookup problems when using
            // nested type inside friend function
            std::map<boost::uint64_t, detail::ptr_helper_ptr>::iterator
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
