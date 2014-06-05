//  Copyright (c) 2011 Thomas Heller
//  Copyright (c) 2013 Hartmut Kaiser
//  Copyright (c) 2014 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_DETAIL_VTABLE_SERIALIZABLE_VTABLE_HPP
#define HPX_UTIL_DETAIL_VTABLE_SERIALIZABLE_VTABLE_HPP

#include <hpx/config/forceinline.hpp>
#include <hpx/util/detail/vtable/vtable.hpp>

namespace hpx { namespace util { namespace detail
{
    template <typename IArchive, typename OArchive>
    struct serializable_vtable
    {
        template <typename T>
        static void save_object(void* const* v, OArchive& ar, unsigned version)
        {
            ar << vtable::get<T>(v);
        }
        typedef void (*save_object_t)(void* const*, OArchive&, unsigned);

        template <typename T>
        static void load_object(void** v, IArchive& ar, unsigned version)
        {
            vtable::default_construct<T>(v);
            ar >> vtable::get<T>(v);
        }
        typedef void (*load_object_t)(void**, IArchive&, unsigned);
    };
}}}

#endif
