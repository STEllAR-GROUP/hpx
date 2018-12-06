//  Copyright (c) 2011 Thomas Heller
//  Copyright (c) 2013 Hartmut Kaiser
//  Copyright (c) 2014-2015 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_DETAIL_VTABLE_VTABLE_HPP
#define HPX_UTIL_DETAIL_VTABLE_VTABLE_HPP

#include <hpx/config.hpp>

#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>

namespace hpx { namespace util { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    struct construct_vtable {};

    template <typename VTable, typename T>
    struct vtables
    {
        static VTable const instance;
    };

    template <typename VTable, typename T>
    VTable const vtables<VTable, T>::instance = construct_vtable<T>();

    template <typename VTable, typename T>
    HPX_CONSTEXPR inline VTable const* get_vtable() noexcept
    {
        static_assert(
            std::is_same<T, typename std::decay<T>::type>::value,
            "T shall have no cv-ref-qualifiers");

        return &vtables<VTable, T>::instance;
    }

    ///////////////////////////////////////////////////////////////////////////
    struct vtable
    {
        static const std::size_t function_storage_size = 3*sizeof(void*);

        template <typename T>
        HPX_FORCEINLINE static T& get(void** v)
        {
            if (sizeof(T) <= function_storage_size)
            {
                return *reinterpret_cast<T*>(v);
            } else {
                return **reinterpret_cast<T**>(v);
            }
        }

        template <typename T>
        HPX_FORCEINLINE static T const& get(void* const* v)
        {
            if (sizeof(T) <= function_storage_size)
            {
                return *reinterpret_cast<T const*>(v);
            } else {
                return **reinterpret_cast<T* const*>(v);
            }
        }

        template <typename T>
        HPX_FORCEINLINE static void default_construct(void** v)
        {
            if (sizeof(T) <= function_storage_size)
            {
                ::new (static_cast<void*>(v)) T; //-V206
            } else {
                *v = new T;
            }
        }

        template <typename T, typename Arg>
        HPX_FORCEINLINE static void construct(void** v, Arg&& arg)
        {
            if (sizeof(T) <= function_storage_size)
            {
                ::new (static_cast<void*>(v)) T(std::forward<Arg>(arg)); //-V206
            } else {
                *v = new T(std::forward<Arg>(arg));
            }
        }

        template <typename T, typename Arg>
        HPX_FORCEINLINE static void reconstruct(void** v, Arg&& arg)
        {
            _delete<T>(v);
            construct<T, Arg>(v, std::forward<Arg>(arg));
        }

        template <typename T>
        HPX_FORCEINLINE static void _destruct(void** v)
        {
            get<T>(v).~T();
        }
        void (*destruct)(void**);

        template <typename T>
        HPX_FORCEINLINE static void _delete(void** v)
        {
            if (sizeof(T) <= function_storage_size)
            {
                _destruct<T>(v);
            } else {
                delete &get<T>(v);
            }
        }
        void (*delete_)(void**);

        template <typename T>
        HPX_CONSTEXPR vtable(construct_vtable<T>) noexcept
          : destruct(&vtable::template _destruct<T>)
          , delete_(&vtable::template _delete<T>)
        {}
    };
}}}

#endif
