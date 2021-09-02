//  Copyright (c) 2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/local/config.hpp>
#include <hpx/serialization/detail/non_default_constructible.hpp>
#include <hpx/serialization/serialization_fwd.hpp>

#include <memory>
#include <type_traits>
#include <utility>

#include <hpx/local/config/warnings_prefix.hpp>

namespace hpx { namespace serialization { namespace detail {

    template <typename T>
    class constructor_selector
    {
    public:
        static T create(input_archive& ar)
        {
            return create(ar, std::is_default_constructible<T>());
        }

        // is default-constructible
        static T create(input_archive& ar, std::true_type)
        {
            T t;
            ar >> t;
            return t;
        }

        // is non-default-constructible
        static T create(input_archive& ar, std::false_type)
        {
            using storage_type =
                typename std::aligned_storage<sizeof(T), alignof(T)>::type;

            storage_type storage;
            T* t = reinterpret_cast<T*>(&storage);
            load_construct_data(ar, t, 0);

            ar >> *t;

            return std::move(*t);
        }
    };
}}}    // namespace hpx::serialization::detail

#include <hpx/local/config/warnings_suffix.hpp>
