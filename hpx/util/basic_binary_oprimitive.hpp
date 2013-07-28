// Copyright (c) 2007-2012 Hartmut Kaiser
// (C) Copyright 2002 Robert Ramey - http://www.rrsd.com .
//
// Use, modification and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#ifndef BASIC_BINARY_OPRIMITIVE_HPP
#define BASIC_BINARY_OPRIMITIVE_HPP

// MS compatible compilers support #pragma once
#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif

// archives stored as native binary - this should be the fastest way
// to archive the state of a group of objects.  It makes no attempt to
// convert to any canonical form.

// IN GENERAL, ARCHIVES CREATED WITH THIS CLASS WILL NOT BE READABLE
// ON PLATFORM APART FROM THE ONE THEY ARE CREATE ON

#include <iosfwd>
#include <string>
#include <cstring> // memcpy
#include <cstddef> // size_t
#include <vector>

#include <boost/assert.hpp>
#include <boost/config.hpp>
#include <boost/cstdint.hpp>
#include <boost/integer.hpp>
#include <boost/integer_traits.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

#include <boost/serialization/is_bitwise_serializable.hpp>
#include <boost/mpl/placeholders.hpp>
#include <boost/serialization/array.hpp>

#include <boost/archive/detail/abi_prefix.hpp> // must be the last header

#include <hpx/util/binary_filter.hpp>

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
    namespace detail
    {
        struct erase_container_type
        {
            virtual ~erase_container_type() {}
            virtual void set_filter(binary_filter* filter) = 0;
            virtual void save_binary(void const* address, std::size_t count) = 0;
        };

        template <typename Container>
        struct container_type : erase_container_type
        {
            container_type(Container& cont)
              : cont_(cont), current_(0), start_compressing_at_(0), filter_(0)
            {}
            ~container_type()
            {
                if (filter_) {
                    std::size_t written = 0;

                    if (cont_.size() < current_)
                        cont_.resize(current_);
                    current_ = start_compressing_at_;

                    do {
                        bool flushed = filter_->flush(&cont_[current_],
                            cont_.size()-current_, written);

                        current_ += written;
                        if (flushed)
                            break;

                        // resize container
                        cont_.resize(cont_.size()*2);

                    } while (true);

                    cont_.resize(current_);         // truncate container
                }
            }

            void set_filter(binary_filter* filter)
            {
                BOOST_ASSERT(0 == filter_);
                filter_ = filter;
                start_compressing_at_ = current_;
            }

            void save_binary(void const* address, std::size_t count)
            {
                BOOST_ASSERT(count != 0);
                {
                    if (filter_) {
                        filter_->save(address, count);
                    }
                    else {
                        if (cont_.size() < current_ + count)
                            cont_.resize(cont_.size() + count);

                        if (count == 1)
                            cont_[current_] = *static_cast<unsigned char const*>(address);
                        else
                            std::memcpy(&cont_[current_], address, count);
                    }
                    current_ += count;
                }
            }

            Container& cont_;
            std::size_t current_;
            std::size_t start_compressing_at_;
            binary_filter* filter_;
        };
    }

    /////////////////////////////////////////////////////////////////////
    // class basic_binary_oprimitive - binary output of primitives to a
    // character buffer
    template <typename Archive>
    class HPX_SERIALIZATION_EXPORT basic_binary_oprimitive
    {
    protected:
        void save_binary(void const* address, std::size_t count)
        {
            size_ += count;
            buffer_->save_binary(address, count);
        }

#ifndef BOOST_NO_MEMBER_TEMPLATE_FRIENDS
        friend class save_access;
    protected:
#else
    public:
#endif
        // this is the output buffer
        std::size_t size_;
        boost::shared_ptr<detail::erase_container_type> buffer_;

        // return a pointer to the most derived class
        Archive* This()
        {
            return static_cast<Archive *>(this);
        }

        // default saving of primitives.
        template <typename T>
        void save(T const& t)
        {
            save_binary(&t, sizeof(T));
        }

        // Trap usage of invalid uninitialized boolean which would otherwise crash
        // on load.
        void save(bool const& t)
        {
            BOOST_ASSERT(0 == static_cast<int>(t) || 1 == static_cast<int>(t));
            save_binary(&t, sizeof(t));
        }

        void save(std::string const& s)
        {
            std::size_t l = s.size();
            This()->save(l);
            if (l != 0)
                save_binary(s.data(), l);
        }
#ifndef BOOST_NO_STD_WSTRING
        void save(std::wstring const& s)
        {
            std::size_t l = s.size();
            This()->save(l);
            if (l != 0)
                save_binary(s.data(), l * sizeof(wchar_t) / sizeof(char));
        }
#endif

        HPX_ALWAYS_EXPORT void init(unsigned flags);

        template <typename Container>
        basic_binary_oprimitive(Container& buffer, unsigned flags = 0)
          : size_(0),
            buffer_(boost::make_shared<detail::container_type<Container> >(buffer))
        {
            init(flags);
        }

    public:
        // we provide an optimized save for all fundamental types
        // typedef serialization::is_bitwise_serializable<mpl::_1>
        // use_array_optimization;
        // workaround without using mpl lambdas
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

        std::size_t bytes_written() const
        {
            return size_;
        }

    protected:
        // the optimized save_array dispatches to save_binary
        template <typename T>
        void save_array(boost::serialization::array<T> const& a, unsigned int)
        {
            save_binary(a.address(), a.count()*sizeof(T));
        }

        void set_filter(binary_filter* filter)
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

