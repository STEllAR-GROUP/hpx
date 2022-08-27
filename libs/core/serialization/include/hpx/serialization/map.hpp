//  Copyright (c) 2015 Anton Bikineev
//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config/endian.hpp>
#include <hpx/assert.hpp>
#include <hpx/serialization/serialization_fwd.hpp>
#include <hpx/serialization/serialize.hpp>
#include <hpx/serialization/traits/is_bitwise_serializable.hpp>
#include <hpx/serialization/traits/is_not_bitwise_serializable.hpp>

#include <cstddef>
#include <cstdint>
#include <map>
#include <type_traits>
#include <utility>

namespace hpx::traits {

    template <typename Key, typename Value>
    struct is_bitwise_serializable<std::pair<Key, Value>>
      : std::integral_constant<bool,
            is_bitwise_serializable_v<std::remove_const_t<Key>> &&
                is_bitwise_serializable_v<std::remove_const_t<Value>>>
    {
    };

    template <typename Key, typename Value>
    struct is_not_bitwise_serializable<std::pair<Key, Value>>
      : std::integral_constant<bool,
            !is_bitwise_serializable_v<std::pair<Key, Value>>>
    {
    };
}    // namespace hpx::traits

namespace hpx::serialization {

    template <typename Key, typename Value>
    void serialize(input_archive& ar, std::pair<Key, Value>& t, unsigned)
    {
        using pair_type = std::pair<Key, Value>;

        constexpr bool optimized =
            hpx::traits::is_bitwise_serializable_v<pair_type> ||
            !hpx::traits::is_not_bitwise_serializable_v<pair_type>;

        if constexpr (optimized)
        {
#if !defined(HPX_SERIALIZATION_HAVE_ALL_TYPES_ARE_BITWISE_SERIALIZABLE)
            if (ar.disable_array_optimization() || ar.endianess_differs())
            {
                ar >>
                    const_cast<
                        std::add_lvalue_reference_t<std::remove_const_t<Key>>>(
                        t.first);
                ar >> t.second;
                return;
            }
#else
            HPX_ASSERT(
                !(ar.disable_array_optimization() || ar.endianess_differs()));
#endif
            load_binary(ar, &t, sizeof(std::pair<Key, Value>));
        }
        else
        {
            ar >> const_cast<
                      std::add_lvalue_reference_t<std::remove_const_t<Key>>>(
                      t.first);
            ar >> t.second;
        }
    }

    template <typename Key, typename Value>
    void serialize(output_archive& ar, std::pair<Key, Value> const& t, unsigned)
    {
        using pair_type = std::pair<Key, Value>;

        constexpr bool optimized =
            hpx::traits::is_bitwise_serializable_v<pair_type> ||
            !hpx::traits::is_not_bitwise_serializable_v<pair_type>;

        if constexpr (optimized)
        {
#if !defined(HPX_SERIALIZATION_HAVE_ALL_TYPES_ARE_BITWISE_SERIALIZABLE)
            if (ar.disable_array_optimization() || ar.endianess_differs())
            {
                ar << t.first << t.second;
                return;
            }
#else
            HPX_ASSERT(
                !(ar.disable_array_optimization() || ar.endianess_differs()));
#endif
            save_binary(ar, &t, sizeof(std::pair<Key, Value>));
        }
        else
        {
            ar << t.first << t.second;
        }
    }

    template <typename Key, typename Value, typename Comp, typename Alloc>
    void serialize(
        input_archive& ar, std::map<Key, Value, Comp, Alloc>& t, unsigned)
    {
        using value_type =
            typename std::map<Key, Value, Comp, Alloc>::value_type;

        std::uint64_t size;
        ar >> size;    //-V128

        t.clear();
        for (std::size_t i = 0; i < size; ++i)
        {
            value_type v;
            ar >> v;
            t.insert(t.end(), HPX_MOVE(v));
        }
    }

    template <typename Key, typename Value, typename Comp, typename Alloc>
    void serialize(output_archive& ar,
        std::map<Key, Value, Comp, Alloc> const& t, unsigned)
    {
        using value_type =
            typename std::map<Key, Value, Comp, Alloc>::value_type;

        std::uint64_t size = t.size();
        ar << size;
        for (value_type const& val : t)
        {
            ar << val;
        }
    }
}    // namespace hpx::serialization
