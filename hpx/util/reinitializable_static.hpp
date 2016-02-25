//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2006 Joao Abecasis
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_REINITIALIZABLE_STATIC_OCT_25_2012_1129AM)
#define HPX_UTIL_REINITIALIZABLE_STATIC_OCT_25_2012_1129AM

#include <hpx/config.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/static_reinit.hpp>

#include <boost/noncopyable.hpp>
#include <boost/call_traits.hpp>
#include <boost/aligned_storage.hpp>

#include <boost/utility/enable_if.hpp>
#include <boost/utility/addressof.hpp>

#include <boost/type_traits/add_pointer.hpp>
#include <boost/type_traits/alignment_of.hpp>

#include <boost/thread/once.hpp>
#include <boost/bind.hpp>

#include <memory>   // for placement new

#if !defined(HPX_WINDOWS)
#  define HPX_EXPORT_REINITIALIZABLE_STATIC HPX_EXPORT
#else
#  define HPX_EXPORT_REINITIALIZABLE_STATIC
#endif

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    //  Provides thread-safe initialization of a single static instance of T.
    //
    //  This instance is guaranteed to be constructed on static storage in a
    //  thread-safe manner, on the first call to the constructor of static_.
    //
    //  Requirements:
    //      T is default constructible or has one argument
    //      T::T() MUST not throw!
    //          this is a requirement of boost::call_once.
    //
    //  In addition this type registers global construction and destruction
    //  functions used by the HPX runtime system to reinitialize the held data
    //  structures.
    template <typename T, typename Tag = T, std::size_t N = 1>
    struct HPX_EXPORT_REINITIALIZABLE_STATIC reinitializable_static;

    //////////////////////////////////////////////////////////////////////////
    template <typename T, typename Tag, std::size_t N>
    struct HPX_EXPORT_REINITIALIZABLE_STATIC reinitializable_static
      : private boost::noncopyable
    {
    public:
        typedef T value_type;

    private:
        static void default_construct()
        {
            for (std::size_t i = 0; i < N; ++i)
                new (get_address(i)) value_type();
        }

        template <typename U>
        static void value_construct(U const& v)
        {
            for (std::size_t i = 0; i < N; ++i)
                new (get_address(i)) value_type(v);
        }

        static void destruct()
        {
            for (std::size_t i = 0; i < N; ++i)
                get_address(i)->~value_type();
        }

        ///////////////////////////////////////////////////////////////////////
        static void default_constructor()
        {
            default_construct();
            reinit_register(
                &reinitializable_static::default_construct,
                &reinitializable_static::destruct);
        }

        template <typename U>
        static void value_constructor(U const* pv)
        {
            value_construct(*pv);
            reinit_register(boost::bind(
                &reinitializable_static::template value_construct<U>, *pv),
                &reinitializable_static::destruct);
        }

    public:
        typedef typename boost::call_traits<T>::reference reference;
        typedef typename boost::call_traits<T>::const_reference const_reference;

        reinitializable_static()
        {
            // do not rely on ADL to find the proper call_once
            boost::call_once(constructed_,
                &reinitializable_static::default_constructor);
        }

        template <typename U>
        reinitializable_static(U const& val)
        {
            // do not rely on ADL to find the proper call_once
            boost::call_once(constructed_,
                boost::bind(
                    &reinitializable_static::template value_constructor<U>,
                    const_cast<U const *>(boost::addressof(val))));
        }

        operator reference()
        {
            return this->get();
        }

        operator const_reference() const
        {
            return this->get();
        }

        reference get(std::size_t item = 0)
        {
            return *this->get_address(item);
        }

        const_reference get(std::size_t item = 0) const
        {
            return *this->get_address(item);
        }

    private:
        typedef typename boost::add_pointer<value_type>::type pointer;

        static pointer get_address(std::size_t item)
        {
            HPX_ASSERT(item < N);
            return static_cast<pointer>(data_[item].address());
        }

        typedef boost::aligned_storage<sizeof(value_type),
            boost::alignment_of<value_type>::value> storage_type;

        static storage_type data_[N];
        static boost::once_flag constructed_;
    };

    template <typename T, typename Tag, std::size_t N>
    typename reinitializable_static<T, Tag, N>::storage_type
        reinitializable_static<T, Tag, N>::data_[N];

    template <typename T, typename Tag, std::size_t N>
    boost::once_flag reinitializable_static<
        T, Tag, N>::constructed_ = BOOST_ONCE_INIT;
}}

#undef HPX_EXPORT_REINITIALIZABLE_STATIC

#endif


