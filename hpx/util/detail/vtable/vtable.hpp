//  Copyright (c) 2011 Thomas Heller
//  Copyright (c) 2013 Hartmut Kaiser
//  Copyright (c) 2014-2015 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_DETAIL_VTABLE_VTABLE_HPP
#define HPX_UTIL_DETAIL_VTABLE_VTABLE_HPP

#include <hpx/config.hpp>
#include <hpx/util/assert.hpp>

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
        static HPX_CONSTEXPR_OR_CONST VTable instance =
            detail::construct_vtable<T>();
    };

    template <typename VTable, typename T>
    HPX_CONSTEXPR_OR_CONST VTable vtables<VTable, T>::instance;

    template <typename VTable, typename T>
    HPX_CONSTEXPR VTable const* get_vtable() noexcept
    {
        static_assert(
            !std::is_reference<T>::value,
            "T shall have no ref-qualifiers");

        return &vtables<VTable, T>::instance;
    }

    ///////////////////////////////////////////////////////////////////////////
    struct vtable
    {
        template <typename T>
        HPX_FORCEINLINE static T& get(void* obj)
        {
            HPX_ASSERT(obj);
            return *reinterpret_cast<T*>(obj);
        }

        template <typename T>
        HPX_FORCEINLINE static T const& get(void const* obj)
        {
            HPX_ASSERT(obj);
            return *reinterpret_cast<T const*>(obj);
        }

        template <typename T>
        HPX_FORCEINLINE static T* default_construct(
            void* storage, std::size_t storage_size)
        {
            if (sizeof(T) <= storage_size)
            {
                return ::new (storage) T; //-V206
            } else {
                return new T;
            }
        }

        template <typename T, typename Arg>
        HPX_FORCEINLINE static T* construct(
            void* storage, std::size_t storage_size, Arg&& arg)
        {
            if (sizeof(T) <= storage_size)
            {
                return ::new (storage) T(std::forward<Arg>(arg)); //-V206
            } else {
                return new T(std::forward<Arg>(arg));
            }
        }

        template <typename T>
        HPX_FORCEINLINE static void _destruct(void* obj)
        {
            if (obj == nullptr)
                return;
            get<T>(obj).~T();
        }
        void (*destruct)(void*);

        template <typename T>
        HPX_FORCEINLINE static void _delete(void* obj, std::size_t storage_size)
        {
            if (sizeof(T) <= storage_size)
            {
                _destruct<T>(obj);
            } else {
                delete static_cast<T*>(obj);
            }
        }
        void (*delete_)(void*, std::size_t storage_size);

        template <typename T>
        HPX_CONSTEXPR vtable(construct_vtable<T>) noexcept
          : destruct(&vtable::template _destruct<T>)
          , delete_(&vtable::template _delete<T>)
        {}
    };
}}}

#endif
