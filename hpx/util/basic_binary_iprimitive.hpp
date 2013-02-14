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

#include <boost/archive/archive_exception.hpp>
#include <boost/mpl/placeholders.hpp>
#include <boost/serialization/is_bitwise_serializable.hpp>

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
    /////////////////////////////////////////////////////////////////////////
    // class basic_binary_iprimitive - read serialized objects from a input
    // binary character buffer
    template <typename Archive>
    class HPX_SERIALIZATION_EXPORT basic_binary_iprimitive
    {
    protected:
        void load_binary(void* address, std::size_t count)
        {
            if (current_+count > size_)
            {
                BOOST_THROW_EXCEPTION(
                    boost::archive::archive_exception(
                        boost::archive::archive_exception::input_stream_error,
                        "archive data bstream is too short"));
                return;
            }

            if (count)
            {
                std::memcpy(address, &buffer_[current_], count);
                current_ += count;
            }
        }

#ifndef BOOST_NO_MEMBER_TEMPLATE_FRIENDS
        friend class load_access;
    protected:
#else
    public:
#endif

        char const* buffer_;
        std::size_t size_;
        std::size_t current_;

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

        template <typename Vector>
        basic_binary_iprimitive(Vector const& buffer, unsigned flags = 0)
          : buffer_(buffer.data()), size_(buffer.size()),current_(0)
        {
            init(flags);
        }

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

    protected:
        // the optimized load_array dispatches to load_binary
        template <typename T>
        void load_array(boost::serialization::array<T>& a, unsigned int)
        {
            load_binary(a.address(), a.count()*sizeof(T));
        }
    };
}}

#undef HPX_SERIALIZATION_EXPORT

#if !defined(BOOST_WINDOWS)
  #pragma GCC visibility pop
#endif

#include <boost/archive/detail/abi_suffix.hpp> // pop pragmas

#endif

