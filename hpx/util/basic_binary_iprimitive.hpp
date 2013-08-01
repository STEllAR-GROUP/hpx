// Copyright (c) 2007-2012 Hartmut Kaiser
// (C) Copyright 2002 Robert Ramey - http://www.rrsd.com .
//
// Use, modification and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#ifndef BASIC_BINARY_IPRIMITIVE_HPP
#define BASIC_BINARY_IPRIMITIVE_HPP

// MS compatible compilers support #pragma once
#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif

#if defined(_MSC_VER)
#pragma warning( disable : 4800 )
#endif

// archives stored as native binary - this should be the fastest way
// to archive the state of a group of objects.  It makes no attempt to
// convert to any canonical form.

// IN GENERAL, ARCHIVES CREATED WITH THIS CLASS WILL NOT BE READABLE
// ON PLATFORM APART FROM THE ONE THEY ARE CREATED ON

#include <iosfwd>
#include <cstring>    // std::memcpy
#include <cstddef>    // std::size_t
#include <string>
#include <vector>

#include <boost/assert.hpp>
#include <boost/config.hpp>
#include <boost/cstdint.hpp>
#include <boost/throw_exception.hpp>
#include <boost/integer.hpp>
#include <boost/integer_traits.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

#include <boost/archive/archive_exception.hpp>
#include <boost/mpl/placeholders.hpp>
#include <boost/serialization/is_bitwise_serializable.hpp>

#include <hpx/util/binary_filter.hpp>
#include <hpx/util/ichunk_manager.hpp>

#include <boost/archive/detail/abi_prefix.hpp> // must be the last header

#if !defined(BOOST_WINDOWS)
#  pragma GCC visibility push(default)
#endif

#if defined(BOOST_MSVC) || defined(BOOST_INTEL_WIN)
#  define HPX_SERIALIZATION_EXPORT
#else
#  define HPX_SERIALIZATION_EXPORT HPX_ALWAYS_EXPORT
#endif

#if defined(BOOST_MSVC)
#  include <intrin.h>
#  pragma intrinsic(memcpy)
#  pragma intrinsic(memset)
#endif

namespace hpx { namespace util
{
//#if BOOST_VERSION < 105500
    namespace dirty_trick
    {
        ///////////////////////////////////////////////////////////////////////
        // Sorry, we need to play this dirty trick because of a bug in
        // boost::serialization::array::operator=() which is missing a return
        // statement prior to Boost V1.55.
        //
        // The method of accessing a private member of a class used is
        // explained here:
        // http://bloglitb.blogspot.com/2010/07/access-to-private-members-thats-easy.html
        //
        template <typename Tag>
        struct result
        {
            typedef typename Tag::type type;
            static type ptr;
        };

        template <typename Tag>
        typename result<Tag>::type result<Tag>::ptr;

        template <typename Tag, typename Tag::type p>
        struct gain_private_access : result<Tag> 
        {
            gain_private_access() { result<Tag>::ptr = p; }
            static gain_private_access filler_object_;
        };

        template <typename Tag, typename Tag::type p>
        typename gain_private_access<Tag, p>
            gain_private_access<Tag, p>::filler_object_;

        ///////////////////////////////////////////////////////////////////////
        template <typename T>
        struct array
        {
            typedef T* boost::serialization::array<T>::* type;
        };

        // Force instantiation of proper template, we instantiate a helper
        // for accessing boost::serialization::array<int>, assuming that all
        // boost::serialization::array<T> have the same structure.
        //
        // Explicit instantiation; the only place where it is legal to pass
        // the address of a private member.  Generates the static ::filler_object_
        // that in turn initializes result<Tag>::type.
        template gain_private_access<
            array<int>, &boost::serialization::array<int>::m_t>;
    }
//#endif

    /////////////////////////////////////////////////////////////////////////--
    // class basic_binary_iprimitive - read serialized objects from a input
    // binary character buffer
    template <typename Archive>
    class HPX_SERIALIZATION_EXPORT basic_binary_iprimitive
    {
    protected:
        void load_binary(void* address, std::size_t count)
        {
            if (0 == count) return;

            buffer_->load_binary(address, count);
            size_ += count;
        }

        void load_binary_chunk(void*& address, std::size_t count)
        {
            if (0 == count) return;

            buffer_->load_binary_chunk(address);
            size_ += count;
        }

#ifndef BOOST_NO_MEMBER_TEMPLATE_FRIENDS
        friend class load_access;
    protected:
#else
    public:
#endif

        boost::uint32_t flags_;
        std::size_t size_;
        boost::shared_ptr<detail::erase_icontainer_type> buffer_;

        // return a pointer to the most derived class
        Archive* This()
        {
            return static_cast<Archive *>(this);
        }

        // main template for serialization of primitive types
        template <typename T>
        void load(T& t)
        {
            load_binary(&t, sizeof(T));
        }

        void load(bool & t)
        {
            load_binary(&t, sizeof(t));
            int i = t;
            BOOST_ASSERT(0 == i || 1 == i);
            (void)i;    // warning suppression for release builds.
        }

        void load(std::string& s)
        {
            std::size_t l;
            This()->load(l);

#if BOOST_WORKAROUND(_RWSTD_VER, BOOST_TESTED_AT(20101))
            if(NULL != s.data())    // borland de-allocator fixup
#endif
                s.resize(l);

            // note breaking a rule here - could be a problem on some platform
            if (l != 0)
                load_binary(&(*s.begin()), l);
        }
#ifndef BOOST_NO_STD_WSTRING
        void load(std::wstring &ws)
        {
            std::size_t l;
            This()->load(l);

#if BOOST_WORKAROUND(_RWSTD_VER, BOOST_TESTED_AT(20101))
            if(NULL != ws.data())   // borland de-allocator fixup
#endif
                ws.resize(l);

            // note breaking a rule here - is could be a problem on some platform
            if (l != 0)
            {
                load_binary(const_cast<wchar_t *>(ws.data()),
                    l * sizeof(wchar_t) / sizeof(char));
            }
        }
#endif

        HPX_ALWAYS_EXPORT void init(unsigned flags);

        template <typename Container>
        basic_binary_iprimitive(Container const& buffer,
                boost::uint64_t inbound_data_size)
          : flags_(0),
            size_(0),
            buffer_(boost::make_shared<detail::icontainer_type<Container> >(
                buffer, inbound_data_size))
        {}

        template <typename Container>
        basic_binary_iprimitive(Container const& buffer, std::vector<chunk>* chunks,
                boost::uint64_t inbound_data_size)
          : flags_(0),
            size_(0),
            buffer_(boost::make_shared<detail::icontainer_type<Container> >(
                buffer, chunks, inbound_data_size))
        {}

    public:
        // we provide an optimized load for all fundamental types
        // typedef serialization::is_bitwise_serializable<mpl::_1>
        // use_array_optimization;
        struct use_array_optimization
        {
    #if defined(BOOST_NO_DEPENDENT_NESTED_DERIVATIONS)
            template <typename T>
            struct apply
            {
                typedef typename
                    boost::serialization::is_bitwise_serializable<T>::type
                type;
            };
    #else
            template <typename T>
            struct apply
              : public boost::serialization::is_bitwise_serializable<T>
            {};
    #endif
        };

        std::size_t bytes_read() const
        {
            return size_;
        }

        boost::uint32_t flags() const
        {
            return flags_;
        }
        void set_flags(boost::uint32_t flags)
        {
            flags_ = flags;
            init(flags_);
        }

    protected:
        // the optimized load_array dispatches to load_binary
        template <typename T>
        void load_array(boost::serialization::array<T>& a)
        {
            if (flags() & disable_data_chunking) {
                load_binary(a.address(), a.count()*sizeof(T));
            }
            else {
                void* address = 0;
                load_binary_chunk(address, a.count()*sizeof(T));
//#if BOOST_VERSION < 105500
                reinterpret_cast<boost::serialization::array<int>&>(a).*
                    dirty_trick::result<dirty_trick::array<int> >::ptr =
                        static_cast<int*>(address);
//#else
//                a = boost::serialization::array<T>(static_cast<T*>(address), a.count());
//#endif
            }
        }

        void set_filter(util::binary_filter* filter)
        {
            buffer_->set_filter(filter);
        }
    };
}}

#undef HPX_SERIALIZATION_EXPORT

#if !defined(BOOST_WINDOWS)
  #pragma GCC visibility pop
#endif

#include <boost/archive/detail/abi_suffix.hpp> // pop pragmas

#endif

