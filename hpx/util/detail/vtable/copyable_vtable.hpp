//  Copyright (c) 2011 Thomas Heller
//  Copyright (c) 2013 Hartmut Kaiser
//  Copyright (c) 2014 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_DETAIL_VTABLE_COPYABLE_VTABLE_HPP
#define HPX_UTIL_DETAIL_VTABLE_COPYABLE_VTABLE_HPP

#include <hpx/config/forceinline.hpp>
#include <hpx/util/detail/vtable/vtable.hpp>

namespace hpx { namespace util { namespace detail
{
    struct copyable_vtable : vtable
    {
        template <typename T>
        HPX_FORCEINLINE static void copy(void** v, void* const* src)
        {
            if (sizeof(T) <= HPX_FUNCTION_STORAGE_NUM_POINTERS*sizeof(void*))
            {
                new (v) T(get<T>(src));
            } else {
                *v = new T(get<T>(src));
            }
        }
        typedef void (*copy_t)(void**, void* const*);
    };
}}}

#endif
