//  Copyright (c) 2014 Thomas Heller
//  Copyright (c) 2015 Anton Bikineev
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// hpxinspect:nodeprecatedinclude:boost/cstdint.hpp

#ifndef HPX_SERIALIZATION_OUTPUT_ARCHIVE_HPP
#define HPX_SERIALIZATION_OUTPUT_ARCHIVE_HPP

#include <hpx/config.hpp>
#include <hpx/runtime/naming_fwd.hpp>
#include <hpx/runtime/serialization/basic_archive.hpp>
#include <hpx/runtime/serialization/detail/polymorphic_nonintrusive_factory.hpp>
#include <hpx/runtime/serialization/detail/raw_ptr.hpp>
#include <hpx/runtime/serialization/output_container.hpp>
#include <hpx/traits/future_access.hpp>
#include <hpx/traits/is_bitwise_serializable.hpp>

#include <boost/cstdint.hpp>

#include <cstddef>
#include <cstdint>
#include <list>
#include <map>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace serialization
{
    struct HPX_EXPORT output_archive
      : basic_archive<output_archive>
    {
        typedef basic_archive<output_archive> base_type;

        typedef std::map<const naming::gid_type*, naming::gid_type> split_gids_type;

        template <typename Container>
        output_archive(Container & buffer,
            std::uint32_t flags = 0U,
            std::vector<serialization_chunk>* chunks = nullptr,
            binary_filter* filter = nullptr)
            : base_type(flags)
            , buffer_(new output_container<Container>(buffer, chunks, filter))
        {
            // endianness needs to be saves separately as it is needed to
            // properly interpret the flags

            // FIXME: make bool once integer compression is implemented
            std::uint64_t endianess = this->base_type::endian_big() ? ~0ul : 0ul;
            save(endianess);

            // send flags sent by the other end to make sure both ends have
            // the same assumptions about the archive format
            save(flags);

            bool has_filter = filter != nullptr;
            save(has_filter);

            if (has_filter && enable_compression())
            {
                *this << detail::raw_ptr(filter);
                buffer_->set_filter(filter);
            }
        }

        void set_split_gids(split_gids_type& split_gids)
        {
            split_gids_ = &split_gids;
        }

        bool is_preprocessing() const
        {
            return buffer_->is_preprocessing();
        }

        template <typename Future>
        void await_future(Future const & f)
        {
            buffer_->await_future(
                *hpx::traits::future_access<Future>::get_shared_state(f));
        }

        std::size_t bytes_written() const
        {
            return size_;
        }

        void add_gid(naming::gid_type const & gid,
            naming::gid_type const & split_gid);

        bool has_gid(naming::gid_type const & gid);

        naming::gid_type get_new_gid(naming::gid_type const & gid);

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
            pointer_tracker_.clear();
            basic_archive<output_archive>::reset();
        }

        void flush()
        {
            buffer_->flush();
        }

    private:
        friend struct basic_archive<output_archive>;
        template <class T>
        friend class array;

        template <typename T>
        void invoke_impl(T const & t)
        {
            save(t);
        }

        template <typename T>
        typename std::enable_if<
            !std::is_integral<T>::value && !std::is_enum<T>::value
        >::type
        save(T const & t)
        {
            typedef hpx::traits::is_bitwise_serializable<T> use_optimized;

            save_bitwise(t, use_optimized());
        }

        template <typename T>
        typename std::enable_if<
            std::is_integral<T>::value || std::is_enum<T>::value
        >::type
        save(T t) //-V659
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
        void save_bitwise(T const & t, std::false_type)
        {
            save_nonintrusively_polymorphic(t,
                hpx::traits::is_nonintrusive_polymorphic<T>());
        }

        // FIXME: think about removing this commented stuff below
        // and adding new free function save_bitwise
        template <typename T>
        void save_bitwise(T const & t, std::true_type)
        {
            static_assert(!std::is_abstract<T>::value,
                "Can not bitwise serialize a class that is abstract");
            if(disable_array_optimization())
            {
                access::serialize(*this, t, 0);
            }
            else
            {
                save_binary(&t, sizeof(t));
            }
        }

        template <typename T>
        void save_nonintrusively_polymorphic(T const & t, std::false_type)
        {
            access::serialize(*this, t, 0);
        }

        template <typename T>
        void save_nonintrusively_polymorphic(T const & t, std::true_type)
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

#if defined(BOOST_HAS_INT128) && !defined(__NVCC__) && \
    !defined(__CUDACC__)
        void save_integral(boost::int128_type t, std::false_type)
        {
            save_integral_impl(t);
        }

        void save_integral(boost::uint128_type t, std::true_type)
        {
            save_integral_impl(t);
        }

        // On some platforms (gcc) std::is_integral<int128>::value
        // evaluates to false. Thus these functions re-route the
        // serialization for those types to the proper implementation
        void save_bitwise(boost::int128_type t, std::false_type)
        {
            save_integral_impl(t);
        }

        void save_bitwise(boost::int128_type t, std::true_type)
        {
            save_integral_impl(t);
        }

        void save_bitwise(boost::uint128_type t, std::false_type)
        {
            save_integral_impl(t);
        }

        void save_bitwise(boost::uint128_type t, std::true_type)
        {
            save_integral_impl(t);
        }
#endif

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

        typedef std::map<const void *, std::uint64_t> pointer_tracker;

        // FIXME: make this function capable for ADL lookup and hence if used
        // as a dependent name it doesn't require output_archive to be complete
        // type or itself to be forwarded
        friend std::uint64_t track_pointer(output_archive& ar, const void* pos)
        {
            pointer_tracker::iterator it = ar.pointer_tracker_.find(pos);
            if(it == ar.pointer_tracker_.end())
            {
                ar.pointer_tracker_.insert(std::make_pair(pos, ar.size_));
                return npos;
            }
            return it->second;
        }

        std::unique_ptr<erased_output_container> buffer_;
        pointer_tracker pointer_tracker_;
        split_gids_type * split_gids_;
    };
}}

#include <hpx/config/warnings_suffix.hpp>

#endif
