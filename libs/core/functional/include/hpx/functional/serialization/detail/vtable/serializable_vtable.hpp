//  Copyright (c) 2011 Thomas Heller
//  Copyright (c) 2013-2022 Hartmut Kaiser
//  Copyright (c) 2014-2015 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/functional/detail/vtable/vtable.hpp>
#include <hpx/serialization/serialization_fwd.hpp>

#include <cstddef>
#include <new>

namespace hpx::util::detail {

    struct serializable_vtable
    {
        template <typename T>
        static void _save_object(void const* obj,
            serialization::output_archive& ar, unsigned /*version*/)
        {
            ar << vtable::get<T>(obj);
        }
        void (*save_object)(
            void const*, serialization::output_archive&, unsigned);

        template <typename T>
        static void* _load_object(void* storage, std::size_t storage_size,
            serialization::input_archive& ar, unsigned /*version*/)
        {
            void* buffer = vtable::allocate<T>(storage, storage_size);
            void* obj = ::new (buffer) T;
            ar >> vtable::get<T>(obj);
            return obj;
        }
        void* (*load_object)(
            void*, std::size_t, serialization::input_archive&, unsigned);

        template <typename T>
        explicit constexpr serializable_vtable(construct_vtable<T>) noexcept
          : save_object(&serializable_vtable::_save_object<T>)
          , load_object(&serializable_vtable::_load_object<T>)
        {
        }
    };
}    // namespace hpx::util::detail
