//  Copyright (c) 2006 Joao Abecasis
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_STATIC_JUN_12_2008_0934AM)
#define HPX_UTIL_STATIC_JUN_12_2008_0934AM

#include <hpx/hpx_fwd.hpp>

#include <boost/noncopyable.hpp>
#include <boost/call_traits.hpp>
#include <boost/aligned_storage.hpp>

#include <boost/utility/enable_if.hpp>
#include <boost/utility/addressof.hpp>

#include <boost/type_traits/add_pointer.hpp>
#include <boost/type_traits/alignment_of.hpp>

#include <boost/thread/once.hpp>
#include <boost/bind.hpp>
#include <boost/static_assert.hpp>

#include <memory>   // for placement new

namespace hpx { namespace util
{
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
    template <typename T, typename Tag = T, std::size_t N = 1>
    struct HPX_EXPORT static_ : boost::noncopyable
    {
    public:
        typedef T value_type;

    private:
        struct destructor
        {
            ~destructor()
            {
                for (std::size_t i = 0; i < N; ++i)
                    static_::get_address(i)->~value_type();
            }
        };

        struct default_constructor
        {
            static void construct()
            {
                for (std::size_t i = 0; i < N; ++i)
                    new (static_::get_address(i)) value_type();
                static destructor d;
            }
        };

        template <typename U>
        struct copy_constructor
        {
            static void construct(U const* pv)
            {
                for (std::size_t i = 0; i < N; ++i)
                    new (static_::get_address(i)) value_type(*pv);
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

        template <typename U>
        static_(U const& val)
        {
            boost::call_once(constructed_,
                boost::bind(&copy_constructor<U>::construct, boost::addressof(val)));
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
            BOOST_ASSERT(item < N);
            return static_cast<pointer>(data_[item].address());
        }

        typedef boost::aligned_storage<sizeof(value_type),
            boost::alignment_of<value_type>::value> storage_type;

        static storage_type data_[N];
        static boost::once_flag constructed_;
    };

    template <typename T, typename Tag, std::size_t N>
    typename static_<T, Tag, N>::storage_type static_<T, Tag, N>::data_[N];

    template <typename T, typename Tag, std::size_t N>
    boost::once_flag static_<T, Tag, N>::constructed_ = BOOST_ONCE_INIT;
}}

#endif // include guard
