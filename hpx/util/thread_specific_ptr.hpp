////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_BABB0428_2085_4DCF_851A_8819D186835E)
#define HPX_BABB0428_2085_4DCF_851A_8819D186835E

#include <boost/config.hpp>

#include <hpx/config/export_definitions.hpp>

#if !defined(BOOST_WINDOWS)
#  define HPX_EXPORT_THREAD_SPECIFIC_PTR HPX_EXPORT
#else
#  define HPX_EXPORT_THREAD_SPECIFIC_PTR
#endif

// native implementation
#if defined(HPX_HAVE_NATIVE_TLS)

#if !defined(__ANDROID__) && !defined(ANDROID)

#if defined(_GLIBCXX_HAVE_TLS)
#  define HPX_NATIVE_TLS __thread
#elif defined(BOOST_WINDOWS)
#  define HPX_NATIVE_TLS __declspec(thread)
#else
#  error "Native thread local storage is not supported for this platform, please undefine HPX_HAVE_NATIVE_TLS"
#endif

#include <boost/assert.hpp>

namespace hpx { namespace util
{

template <typename T, typename Tag>
struct HPX_EXPORT_THREAD_SPECIFIC_PTR thread_specific_ptr
{
    typedef T element_type;

    T* get() const
    {
        return ptr_;
    }

    T* operator->() const
    {
        return ptr_;
    }

    T& operator*() const
    {
        BOOST_ASSERT(0 != ptr_);
        return *ptr_;
    }

    void reset(
        T* new_value = 0
        )
    {
        if (0 != ptr_) //-V809
            delete ptr_;

        ptr_ = new_value;
    }

  private:
    static HPX_NATIVE_TLS T* ptr_;
};

template <typename T, typename Tag>
HPX_NATIVE_TLS T* thread_specific_ptr<T, Tag>::ptr_ = 0;

}}

#else

#include <pthread.h>
#include <boost/assert.hpp>
#include <hpx/util/static.hpp>

namespace hpx { namespace util
{
    namespace detail
    {
        struct thread_specific_ptr_key
        {
            /*
            static void make_key()
            {
                pthread_key_create(&key, NULL);
            }
            */

            thread_specific_ptr_key()
                //: key_once(PTHREAD_ONCE_INIT)
            {
                //pthread_once(&key_once, &thread_specific_ptr_key::make_key);
                pthread_key_create(&key, NULL);
            }

            pthread_key_t key;
            //pthread_once_t key_once;
        };
    }

template <typename T, typename Tag>
struct HPX_EXPORT_THREAD_SPECIFIC_PTR thread_specific_ptr
{
    typedef T element_type;

    static pthread_key_t get_key()
    {
        static_<detail::thread_specific_ptr_key, thread_specific_ptr<T, Tag> > key_holder;

        return key_holder.get().key;
    }

    T* get() const
    {
        T* ptr = 0;

        ptr = reinterpret_cast<T *>(pthread_getspecific(thread_specific_ptr<T, Tag>::get_key()));

        return ptr;
    }

    T* operator->() const
    {
        T* ptr = 0;

        ptr = reinterpret_cast<T *>(pthread_getspecific(thread_specific_ptr<T, Tag>::get_key()));

        return ptr;
    }

    T& operator*() const
    {
        T* ptr = 0;

        ptr = reinterpret_cast<T *>(pthread_getspecific(thread_specific_ptr<T, Tag>::get_key()));
        BOOST_ASSERT(0 != ptr);
        return *ptr;
    }

    void reset(
        T* new_value = 0
        )
    {
        T* ptr = 0;

        ptr = reinterpret_cast<T *>(pthread_getspecific(thread_specific_ptr<T, Tag>::get_key()));
        if (0 != ptr)
            delete ptr;

        ptr = new_value;
        pthread_setspecific(thread_specific_ptr<T, Tag>::get_key(), ptr);
    }

  private:
};

/*
template <typename T, typename Tag>
static_<detail::thread_specific_ptr_key> thread_specific_ptr<T, Tag>::key_holder = detail::thread_specific_ptr_key();
*/

}}

#endif

// fallback implementation
#else

#include <boost/thread/tss.hpp>

namespace hpx { namespace util
{

template <typename T, typename Tag>
struct HPX_EXPORT_THREAD_SPECIFIC_PTR thread_specific_ptr
{
    typedef typename boost::thread_specific_ptr<T>::element_type element_type;

    T* get() const
    {
        return ptr_.get();
    }

    T* operator->() const
    {
        return ptr_.operator->();
    }

    T& operator*() const
    {
        return ptr_.operator*();
    }

    void reset(
        T* new_value = 0
        )
    {
        ptr_.reset(new_value);
    }

  private:
    static boost::thread_specific_ptr<T> ptr_;
};

template <typename T, typename Tag>
boost::thread_specific_ptr<T> thread_specific_ptr<T, Tag>::ptr_;

}}

#endif

#endif // HPX_BABB0428_2085_4DCF_851A_8819D186835E

