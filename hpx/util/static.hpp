//  Copyright (c) 2006 Joao Abecasis
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_STATIC_JUN_12_2008_0934AM)
#define HPX_UTIL_STATIC_JUN_12_2008_0934AM

#include <hpx/config.hpp>

#include <boost/call_traits.hpp>
#include <boost/aligned_storage.hpp>

#include <boost/utility/enable_if.hpp>
#include <boost/utility/addressof.hpp>

#include <boost/type_traits/add_pointer.hpp>
#include <boost/type_traits/alignment_of.hpp>

#if !defined(HPX_GCC_VERSION) && !defined(HPX_CLANG_VERSION) && \
    !(HPX_INTEL_VERSION > 1200 && !defined(HPX_WINDOWS)) && \
    (_MSC_FULL_VER < 180021114)         // NovCTP_2013
#include <boost/thread/once.hpp>
#include <boost/bind.hpp>

#include <memory>   // for placement new
#endif

#if !defined(HPX_WINDOWS)
#  define HPX_EXPORT_STATIC_ HPX_EXPORT
#else
#  define HPX_EXPORT_STATIC_
#endif

namespace hpx { namespace util
{
#if defined(HPX_GCC_VERSION) || defined(HPX_CLANG_VERSION) || \
    (HPX_INTEL_VERSION > 1200 && !defined(HPX_WINDOWS)) || \
    (_MSC_FULL_VER >= 180021114)         // NovCTP_2013

    //
    // C++11 requires thread-safe initialization of function-scope statics.
    // For conforming compilers, we utilize this feature.
    //
    template <typename T, typename Tag = T>
    struct HPX_EXPORT_STATIC_ static_
    {
    private:
        HPX_NON_COPYABLE(static_);

    public:
        typedef T value_type;

        typedef typename boost::call_traits<T>::reference reference;
        typedef typename boost::call_traits<T>::const_reference const_reference;

        static_()
        {
            get_reference();
        }

        operator reference()
        {
            return get();
        }

        operator const_reference() const
        {
            return get();
        }

        reference get()
        {
            return get_reference();
        }

        const_reference get() const
        {
            return get_reference();
        }

    private:
        static reference get_reference()
        {
            static T t;
            return t;
        }
    };

#else

    //
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
    template <typename T, typename Tag = T>
    struct HPX_EXPORT_STATIC_ static_
    {
    private:
        HPX_NON_COPYABLE(static_);

    public:
        typedef T value_type;

    private:
        struct destructor
        {
            ~destructor()
            {
                static_::get_address()->~value_type();
            }
        };

        struct default_constructor
        {
            static void construct()
            {
                new (static_::get_address()) value_type();
                static destructor d;
            }
        };

    public:
        typedef typename boost::call_traits<T>::reference reference;
        typedef typename boost::call_traits<T>::const_reference const_reference;

        static_()
        {
            boost::call_once(&default_constructor::construct, constructed_);
        }

        operator reference()
        {
            return this->get();
        }

        operator const_reference() const
        {
            return this->get();
        }

        reference get()
        {
            return *this->get_address();
        }

        const_reference get() const
        {
            return *this->get_address();
        }

    private:
        typedef typename boost::add_pointer<value_type>::type pointer;

        static pointer get_address()
        {
            return static_cast<pointer>(data_.address());
        }

        typedef boost::aligned_storage<sizeof(value_type),
            boost::alignment_of<value_type>::value> storage_type;

        static storage_type data_;
        static boost::once_flag constructed_;
    };

    template <typename T, typename Tag>
    typename static_<T, Tag>::storage_type static_<T, Tag>::data_;

    template <typename T, typename Tag>
    boost::once_flag static_<T, Tag>::constructed_ = BOOST_ONCE_INIT;
#endif
}}

#endif // include guard
