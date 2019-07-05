//  Copyright (c) 2011 Thomas Heller
//  Copyright (c) 2013 Hartmut Kaiser
//  Copyright (c) 2014-2015 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_DETAIL_VTABLE_SERIALIZABLE_VTABLE_HPP
#define HPX_UTIL_DETAIL_VTABLE_SERIALIZABLE_VTABLE_HPP

#include <hpx/config.hpp>
#include <hpx/runtime/serialization/serialization_fwd.hpp>
#include <hpx/util/detail/vtable/vtable.hpp>

#include <cstddef>
#include <new>

namespace hpx { namespace util { namespace detail
{
    struct serializable_vtable
    {
        template <typename T>
        static void _save_object(void const* obj,
            serialization::output_archive& ar, unsigned /*version*/)
        {
            ar << vtable::get<T>(obj);
        }
        void (*save_object)(void const*,
            serialization::output_archive&, unsigned);

        template <typename T>
        static void* _load_object(void* storage, std::size_t storage_size,
            serialization::input_archive& ar, unsigned /*version*/)
        {
            void* buffer = vtable::allocate<T>(storage, storage_size);
            void* obj = ::new (buffer) T;
            ar >> vtable::get<T>(obj);
            return obj;
        }
        void* (*load_object)(void*, std::size_t,
            serialization::input_archive&, unsigned);

        template <typename T>
        HPX_CONSTEXPR serializable_vtable(construct_vtable<T>) noexcept
          : save_object(&serializable_vtable::template _save_object<T>)
          , load_object(&serializable_vtable::template _load_object<T>)
        {}
    };
}}}

#endif
