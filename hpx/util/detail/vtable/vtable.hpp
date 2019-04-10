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
#include <type_traits>

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
        static T& get(void* obj) noexcept
        {
            return *reinterpret_cast<T*>(obj);
        }

        template <typename T>
        static T const& get(void const* obj) noexcept
        {
            return *reinterpret_cast<T const*>(obj);
        }

        template <typename T>
        static void* allocate(void* storage, std::size_t storage_size)
        {
            using storage_t =
                typename std::aligned_storage<sizeof(T), alignof(T)>::type;

            if (sizeof(T) > storage_size) {
                return new storage_t;
            }
            return storage;
        }

        template <typename T>
        static void _deallocate(void* obj, std::size_t storage_size, bool destroy)
        {
            using storage_t =
                typename std::aligned_storage<sizeof(T), alignof(T)>::type;

            if (destroy) {
                get<T>(obj).~T();
            }

            if (sizeof(T) > storage_size) {
                delete static_cast<storage_t*>(obj);
            }
        }
        void (*deallocate)(void*, std::size_t storage_size, bool);

        template <typename T>
        HPX_CONSTEXPR vtable(construct_vtable<T>) noexcept
          : deallocate(&vtable::template _deallocate<T>)
        {}
    };
}}}

#endif
