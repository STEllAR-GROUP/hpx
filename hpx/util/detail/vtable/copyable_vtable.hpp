//  Copyright (c) 2011 Thomas Heller
//  Copyright (c) 2013 Hartmut Kaiser
//  Copyright (c) 2014-2015 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_DETAIL_VTABLE_COPYABLE_VTABLE_HPP
#define HPX_UTIL_DETAIL_VTABLE_COPYABLE_VTABLE_HPP

#include <hpx/config.hpp>
#include <hpx/util/detail/vtable/vtable.hpp>

#include <cstddef>
#include <new>

namespace hpx { namespace util { namespace detail
{
    struct copyable_vtable
    {
        template <typename T>
        static void* _copy(
            void* storage, std::size_t storage_size, void const* src, bool destroy)
        {
            if (destroy)
                vtable::get<T>(storage).~T();

            void* buffer = vtable::allocate<T>(storage, storage_size);
            return ::new (buffer) T(vtable::get<T>(src));
        }
        void* (*copy)(void*, std::size_t, void const*, bool);

        HPX_CONSTEXPR copyable_vtable(std::nullptr_t) noexcept
          : copy(nullptr)
        {}

        template <typename T>
        HPX_CONSTEXPR copyable_vtable(construct_vtable<T>) noexcept
          : copy(&copyable_vtable::template _copy<T>)
        {}
    };
}}}

#endif
