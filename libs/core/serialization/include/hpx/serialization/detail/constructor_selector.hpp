//  Copyright (c) 2021-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/serialization/detail/non_default_constructible.hpp>
#include <hpx/serialization/serialization_fwd.hpp>
#include <hpx/serialization/serialize.hpp>

#include <type_traits>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx::serialization::detail {

    template <typename T>
    class constructor_selector
    {
    public:
        static T create(input_archive& ar)
        {
            if constexpr (std::is_default_constructible_v<T>)
            {
                T t;
                ar.load(t);
                return t;
            }
            else
            {
                using storage_type =
                    std::aligned_storage_t<sizeof(T), alignof(T)>;

                storage_type storage;
                T* t = reinterpret_cast<T*>(&storage);
                load_construct_data(ar, t, 0);

                ar.load(*t);

                return *t;
            }
        }
    };
}    // namespace hpx::serialization::detail

#include <hpx/config/warnings_suffix.hpp>
