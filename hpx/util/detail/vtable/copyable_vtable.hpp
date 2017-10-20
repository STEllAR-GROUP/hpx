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

namespace hpx { namespace util { namespace detail
{
    struct copyable_vtable
    {
        template <typename T>
        HPX_FORCEINLINE static void _copy(void** v, void* const* src)
        {
            if (sizeof(T) <= vtable::function_storage_size)
            {
                new (v) T(vtable::get<T>(src));
            } else {
                *v = new T(vtable::get<T>(src));
            }
        }
        void (*copy)(void**, void* const*);

        template <typename T>
        HPX_CONSTEXPR copyable_vtable(construct_vtable<T>) noexcept
          : copy(&copyable_vtable::template _copy<T>)
        {}
    };
}}}

#endif
