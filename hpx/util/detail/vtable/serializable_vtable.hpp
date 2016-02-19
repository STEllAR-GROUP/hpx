//  Copyright (c) 2011 Thomas Heller
//  Copyright (c) 2013 Hartmut Kaiser
//  Copyright (c) 2014 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_DETAIL_VTABLE_SERIALIZABLE_VTABLE_HPP
#define HPX_UTIL_DETAIL_VTABLE_SERIALIZABLE_VTABLE_HPP

#include <hpx/config/forceinline.hpp>
#include <hpx/runtime/serialization/serialization_fwd.hpp>
#include <hpx/util/detail/vtable/vtable.hpp>

namespace hpx { namespace util { namespace detail
{
    struct serializable_vtable
    {
        template <typename T>
        static void save_object(void* const* v,
            serialization::output_archive& ar, unsigned version)
        {
            ar << vtable::get<T>(v);
        }
        typedef void (*save_object_t)(void* const*,
            serialization::output_archive&, unsigned);

        template <typename T>
        static void load_object(void** v,
            serialization::input_archive& ar, unsigned version)
        {
            vtable::default_construct<T>(v);
            ar >> vtable::get<T>(v);
        }
        typedef void (*load_object_t)(void**,
            serialization::input_archive&, unsigned);
    };
}}}

#endif
