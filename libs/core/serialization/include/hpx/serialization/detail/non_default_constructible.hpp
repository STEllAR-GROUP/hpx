//  Copyright (c) 2017 Anton Bikineev
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/serialization/traits/is_not_bitwise_serializable.hpp>

#include <memory>
#include <type_traits>

namespace hpx::serialization::detail {

    ///////////////////////////////////////////////////////////////////////////
    // default fall-backs for serializing non-default-constructible types
    template <typename Archive, typename T,
        std::enable_if_t<traits::is_not_bitwise_serializable_v<T>, int> = 0>
    void save_construct_data(Archive&, T const*, unsigned)
    {
    }

    // by default fall back to in-place default construction
    template <typename Archive, typename T,
        std::enable_if_t<traits::is_not_bitwise_serializable_v<T>, int> = 0>
    void load_construct_data(Archive&, T* t, unsigned)
    {
        ::new (t) T;
    }

#if defined(HPX_SERIALIZATION_HAVE_ALL_TYPES_ARE_BITWISE_SERIALIZABLE)
    template <typename Archive, typename T,
        std::enable_if_t<!traits::is_not_bitwise_serializable_v<T>, int> = 0>
    void save_construct_data(Archive&, T const*, unsigned)
    {
    }

    // by default fall back to in-place default construction
    template <typename Archive, typename T,
        std::enable_if_t<!traits::is_not_bitwise_serializable_v<T>, int> = 0>
    void load_construct_data(Archive&, T* t, unsigned)
    {
        ::new (t) T;
    }
#endif
}    // namespace hpx::serialization::detail
