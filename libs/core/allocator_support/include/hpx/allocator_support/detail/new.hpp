//  Copyright (c) 2015 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/type_support/construct_at.hpp>

#include <utility>

namespace hpx::util::functional {

    template <typename T>
    struct new_
    {
        template <typename... Ts>
        T* operator()(Ts&&... vs) const
        {
            return new T(HPX_FORWARD(Ts, vs)...);
        }
    };

    template <typename T>
    struct placement_new
    {
        template <typename... Ts>
        T* operator()(void* p, Ts&&... vs) const
        {
            return hpx::construct_at(
                static_cast<T*>(p), HPX_FORWARD(Ts, vs)...);
        }
    };

    template <typename T>
    struct placement_new_one
    {
        explicit constexpr placement_new_one(void* p) noexcept
          : p_(p)
        {
        }

        template <typename... Ts>
        T* operator()(Ts&&... vs) const
        {
            return hpx::construct_at(
                static_cast<T*>(p_), HPX_FORWARD(Ts, vs)...);
        }

        void* p_;
    };
}    // namespace hpx::util::functional
