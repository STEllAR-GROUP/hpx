////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_BABB0428_2085_4DCF_851A_8819D186835E)
#define HPX_BABB0428_2085_4DCF_851A_8819D186835E

#include <hpx/config.hpp>
#include <hpx/assertion.hpp>

// Needed to get potentially get _GLIBCXX_HAVE_TLS
#include <cstdlib>

// clang-format off
#if !defined(HPX_WINDOWS)
#  define HPX_EXPORT_THREAD_SPECIFIC_PTR HPX_EXPORT
#else
#  define HPX_EXPORT_THREAD_SPECIFIC_PTR
#endif
// clang-format on

// native implementation
#if defined(HPX_HAVE_NATIVE_TLS)

#if (!defined(__ANDROID__) && !defined(ANDROID)) && !defined(__bgq__)

namespace hpx { namespace util {
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
            HPX_ASSERT(nullptr != ptr_);
            return *ptr_;
        }

        void reset(T* new_value = nullptr)
        {
            delete ptr_;
            ptr_ = new_value;
        }

    private:
        static HPX_NATIVE_TLS T* ptr_;
    };

    template <typename T, typename Tag>
    HPX_NATIVE_TLS T* thread_specific_ptr<T, Tag>::ptr_ = nullptr;
}}    // namespace hpx::util

#else

#include <hpx/type_support/static.hpp>
#include <pthread.h>

namespace hpx { namespace util {
    namespace detail {
        struct thread_specific_ptr_key
        {
            thread_specific_ptr_key()
            {
                //pthread_once(&key_once, &thread_specific_ptr_key::make_key);
                pthread_key_create(&key, nullptr);
            }

            pthread_key_t key;
        };
    }    // namespace detail

    template <typename T, typename Tag>
    struct HPX_EXPORT_THREAD_SPECIFIC_PTR thread_specific_ptr
    {
        typedef T element_type;

        static pthread_key_t get_key()
        {
            static_<detail::thread_specific_ptr_key,
                thread_specific_ptr<T, Tag>>
                key_holder;

            return key_holder.get().key;
        }

        T* get() const
        {
            return reinterpret_cast<T*>(
                pthread_getspecific(thread_specific_ptr<T, Tag>::get_key()));
        }

        T* operator->() const
        {
            return reinterpret_cast<T*>(
                pthread_getspecific(thread_specific_ptr<T, Tag>::get_key()));
        }

        T& operator*() const
        {
            T* ptr = nullptr;

            ptr = reinterpret_cast<T*>(
                pthread_getspecific(thread_specific_ptr<T, Tag>::get_key()));
            HPX_ASSERT(nullptr != ptr);
            return *ptr;
        }

        void reset(T* new_value = nullptr)
        {
            T* ptr = nullptr;

            ptr = reinterpret_cast<T*>(
                pthread_getspecific(thread_specific_ptr<T, Tag>::get_key()));
            if (nullptr != ptr)
                delete ptr;

            ptr = new_value;
            pthread_setspecific(thread_specific_ptr<T, Tag>::get_key(), ptr);
        }
    };
}}    // namespace hpx::util

#endif

// fallback implementation
#else

#include <boost/thread/tss.hpp>

namespace hpx { namespace util {
    template <typename T, typename Tag>
    struct HPX_EXPORT_THREAD_SPECIFIC_PTR thread_specific_ptr
    {
        typedef
            typename boost::thread_specific_ptr<T>::element_type element_type;

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

        void reset(T* new_value = nullptr)
        {
            ptr_.reset(new_value);
        }

    private:
        static boost::thread_specific_ptr<T> ptr_;
    };

    template <typename T, typename Tag>
    boost::thread_specific_ptr<T> thread_specific_ptr<T, Tag>::ptr_;
}}    // namespace hpx::util

#endif

#endif    // HPX_BABB0428_2085_4DCF_851A_8819D186835E
