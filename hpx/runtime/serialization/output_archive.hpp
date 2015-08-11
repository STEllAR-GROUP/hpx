//  Copyright (c) 2014 Thomas Heller
//  Copyright (c) 2015 Anton Bikineev
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_SERIALIZATION_OUTPUT_ARCHIVE_HPP
#define HPX_SERIALIZATION_OUTPUT_ARCHIVE_HPP

#include <hpx/runtime/serialization/basic_archive.hpp>
#include <hpx/runtime/serialization/output_container.hpp>
#include <hpx/runtime/serialization/detail/polymorphic_nonintrusive_factory.hpp>
#include <hpx/runtime/serialization/detail/raw_ptr.hpp>

#include <boost/mpl/or.hpp>
#include <boost/type_traits/is_integral.hpp>
#include <boost/type_traits/is_unsigned.hpp>
#include <boost/type_traits/is_enum.hpp>
#include <boost/utility/enable_if.hpp>

#include <memory>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace serialization
{
    struct HPX_EXPORT output_archive
      : basic_archive<output_archive>
    {
        typedef basic_archive<output_archive> base_type;
        typedef std::map<const void *, boost::uint64_t> pointer_tracker;

        template <typename Container>
        output_archive(Container & buffer,
            boost::uint32_t flags = 0U,
            boost::uint32_t dest_locality_id = ~0U,
            std::vector<serialization_chunk>* chunks = 0,
            binary_filter* filter = 0)
            : base_type(flags)
            , buffer_(new output_container<Container>(buffer, chunks, filter))
            , dest_locality_id_(dest_locality_id)
        {
            // endianness needs to be saves separately as it is needed to
            // properly interpret the flags

            // FIXME: make bool once integer compression is implemented
            boost::uint64_t endianess = this->base_type::endian_big() ? ~0ul : 0ul;
            save(endianess);

            // send flags sent by the other end to make sure both ends have
            // the same assumptions about the archive format
            save(flags);

            bool has_filter = filter != 0;
            save(has_filter);

            if (has_filter && enable_compression())
            {
                *this << detail::raw_ptr(filter);
                buffer_->set_filter(filter);
            }
        }

        bool is_saving() const
        {
            return buffer_->is_saving();
        }

        template <typename T>
        void invoke_impl(T const & t)
        {
            save(t);
        }

        template <typename T>
        typename boost::disable_if<
            boost::mpl::or_<
                boost::is_integral<T>
              , boost::is_enum<T>
            >
        >::type
        save(T const & t)
        {
            save_bitwise(t,
                typename hpx::traits::is_bitwise_serializable<T>::type());
        }

        template <typename T>
        void save_bitwise(T const & t, boost::mpl::false_)
        {
            save_nonintrusively_polymorphic(t,
                hpx::traits::is_nonintrusive_polymorphic<T>());
        }

        //think about remaining this commented stuff below
        //and adding new free function save_bitwise
        template <typename T>
        void save_bitwise(T const & t, boost::mpl::true_)
        {
            BOOST_STATIC_ASSERT_MSG(!boost::is_abstract<T>::value,
                "Can not bitwise serialize a class that is abstract");
            if(disable_array_optimization())
            {
                serialize(*this, const_cast<T &>(t), 0);
            }
            else
            {
                save_binary(&t, sizeof(t));
            }
        }

        template <typename T>
        void save_nonintrusively_polymorphic(T const & t, boost::mpl::false_)
        {
            serialize(*this, const_cast<T &>(t), 0);
        }

        template <typename T>
        void save_nonintrusively_polymorphic(T const & t, boost::mpl::true_)
        {
            detail::polymorphic_nonintrusive_factory::instance().save(*this, t);
        }

        template <typename T>
        typename boost::enable_if<
            boost::mpl::or_<
                boost::is_integral<T>
              , boost::is_enum<T>
            >
        >::type
        save(T t)
        {
            save_integral(t,
                typename boost::is_unsigned<T>::type());
        }

        template <typename T>
        void save_integral(T val, boost::mpl::false_)
        {
            save_integral_impl(static_cast<boost::int64_t>(val));
        }

        template <typename T>
        void save_integral(T val, boost::mpl::true_)
        {
            save_integral_impl(static_cast<boost::uint64_t>(val));
        }

#if defined(BOOST_HAS_INT128)
        void save_integral(boost::int128_type t, boost::mpl::false_)
        {
            save_integral_impl(t);
        }

        void save_integral(boost::uint128_type t, boost::mpl::true_)
        {
            save_integral_impl(t);
        }
#endif

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

        template <class Promoted>
        void save_integral_impl(Promoted l)
        {
            const std::size_t size = sizeof(Promoted);
            char* cptr = reinterpret_cast<char *>(&l); //-V206
#ifdef BOOST_BIG_ENDIAN
            if(endian_little())
                reverse_bytes(size, cptr);
#else
            if(endian_big())
                reverse_bytes(size, cptr);
#endif

            save_binary(cptr, size);
        }

        void save_binary(void const * address, std::size_t count)
        {
            if(count == 0) return;
            size_ += count;
            buffer_->save_binary(address, count);
        }

        void save_binary_chunk(void const * address, std::size_t count)
        {
            if(count == 0) return;
            size_ += count;
            if(disable_data_chunking())
              buffer_->save_binary(address, count);
            else
              buffer_->save_binary_chunk(address, count);
        }

        boost::uint64_t track_pointer(const void * p)
        {
            pointer_tracker::iterator it = pointer_tracker_.find(p);
            if(it == pointer_tracker_.end())
            {
                pointer_tracker_.insert(std::make_pair(p, size_));
                return npos;
            }
            return it->second;
        }

        boost::uint32_t get_dest_locality_id() const
        {
            return dest_locality_id_;
        }

        std::size_t bytes_written() const
        {
            return size_;
        }

    private:
        std::unique_ptr<erased_output_container> buffer_;
        pointer_tracker pointer_tracker_;
        boost::uint32_t dest_locality_id_;
    };

    BOOST_FORCEINLINE
    boost::uint64_t track_pointer(output_archive & ar, const void * pos)
    {
        return ar.track_pointer(pos);
    }
}}

#include <hpx/config/warnings_suffix.hpp>

#endif
