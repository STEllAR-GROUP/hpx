//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c) 2006 Joao Abecasis
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_LOCAL_REINITIALIZABLE_STATIC_JAN_20_2013_0409PM)
#define HPX_LCOS_LOCAL_REINITIALIZABLE_STATIC_JAN_20_2013_0409PM

#include <hpx/config.hpp>
#include <hpx/assertion.hpp>
#include <hpx/lcos/local/once.hpp>
#include <hpx/util/static_reinit.hpp>

#include <cstddef>
#include <memory>   // for placement new
#include <type_traits>

#if !defined(HPX_WINDOWS)
#  define HPX_EXPORT_REINITIALIZABLE_STATIC HPX_EXPORT
#else
#  define HPX_EXPORT_REINITIALIZABLE_STATIC
#endif

namespace hpx { namespace lcos { namespace local
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

    template <typename T, typename Tag, std::size_t N>
    struct HPX_EXPORT_REINITIALIZABLE_STATIC reinitializable_static
    {
    public:
        HPX_NON_COPYABLE(reinitializable_static);

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
            util::reinit_register(
                &reinitializable_static::default_construct,
                &destruct);
        }

        template <typename U>
        static void value_constructor(U const* pv)
        {
            value_construct(*pv);
            util::reinit_register(
                [=]() -> void { value_construct(*pv); },
                &destruct);
        }

    public:
        typedef T & reference;
        typedef T const & const_reference;

        reinitializable_static()
        {
            // do not rely on ADL to find the proper call_once
            lcos::local::call_once(constructed_,
                &reinitializable_static::default_constructor);
        }

        template <typename U>
        reinitializable_static(U const& val)
        {
            // do not rely on ADL to find the proper call_once
            lcos::local::call_once(constructed_, [&]() -> void {
                return value_constructor(std::addressof(val));
            });
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
        typedef typename std::add_pointer<value_type>::type pointer;

        static pointer get_address(std::size_t item)
        {
            HPX_ASSERT(item < N);
            return static_cast<pointer>(data_[item].address());
        }

        typedef typename std::aligned_storage<sizeof(value_type),
            std::alignment_of<value_type>::value>::type storage_type;

        static storage_type data_[N];
        static lcos::local::once_flag constructed_;
    };

    template <typename T, typename Tag, std::size_t N>
    typename reinitializable_static<T, Tag, N>::storage_type
        reinitializable_static<T, Tag, N>::data_[N];

    template <typename T, typename Tag, std::size_t N>
    lcos::local::once_flag reinitializable_static<
            T, Tag, N>::constructed_;
}}}

#undef HPX_EXPORT_REINITIALIZABLE_STATIC

#endif


