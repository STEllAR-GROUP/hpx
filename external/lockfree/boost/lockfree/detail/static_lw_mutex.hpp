//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(BOOST_LOCKFREE_DETAIL_STATIC_LW_MUTEX_JUL_11_2008_0430PM)
#define BOOST_LOCKFREE_DETAIL_STATIC_LW_MUTEX_JUL_11_2008_0430PM

#include <boost/noncopyable.hpp>
#include <boost/thread/once.hpp>
#include <boost/call_traits.hpp>
#include <boost/aligned_storage.hpp>
#include <boost/type_traits/add_pointer.hpp>
#include <boost/type_traits/alignment_of.hpp>

namespace boost { namespace lockfree { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <class T, class Tag>
    struct static_ : boost::noncopyable
    {
        typedef T value_type;
        typedef typename boost::call_traits<T>::reference reference;
        typedef typename boost::call_traits<T>::const_reference const_reference;

    private:
        struct destructor
        {
            ~destructor()
            {
                static_::get_address()->~value_type();
            }
        };

        struct default_ctor
        {
            static void construct()
            {
                ::new (static_::get_address()) value_type();
                static destructor d;
            }
        };
        
    public:
        static_(Tag = Tag())
        {
            boost::call_once(&default_ctor::construct, constructed_);
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

    template <class T, class Tag>
    typename static_<T, Tag>::storage_type static_<T, Tag>::data_;

    template <class T, class Tag>
    boost::once_flag static_<T, Tag>::constructed_ = BOOST_ONCE_INIT;

}}}

#endif

