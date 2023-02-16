//  Copyright (c) 2011 Thomas Heller
//  Copyright (c) 2013-2023 Hartmut Kaiser
//  Copyright (c) 2014-2015 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/functional/detail/vtable/vtable.hpp>

#include <cstddef>
#include <memory>

namespace hpx::util::detail {

    struct copyable_vtable
    {
        template <typename T>
        static void* _copy(void* storage, std::size_t storage_size,
            void const* src, bool destroy)
        {
            if (destroy)
            {
                std::destroy_at(std::addressof(vtable::get<T>(storage)));
            }

            void* buffer = vtable::allocate<T>(storage, storage_size);
            return ::new (buffer) T(vtable::get<T>(src));
        }
        void* (*copy)(void*, std::size_t, void const*, bool) = nullptr;

        explicit constexpr copyable_vtable(std::nullptr_t) noexcept {}

        template <typename T>
        explicit constexpr copyable_vtable(construct_vtable<T>) noexcept
          : copy(&copyable_vtable::_copy<T>)
        {
        }
    };
}    // namespace hpx::util::detail
