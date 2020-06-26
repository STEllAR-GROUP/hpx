//  Copyright (c) 2015 Anton Bikineev
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/serialization/serialization_fwd.hpp>
#include <hpx/serialization/serialize.hpp>
#include <hpx/serialization/traits/is_bitwise_serializable.hpp>

#include <cstddef>
#include <cstdint>
#include <map>
#include <type_traits>
#include <utility>

namespace hpx {

    namespace traits {

        template <typename Key, typename Value>
        struct is_bitwise_serializable<std::pair<Key, Value>>
          : std::integral_constant<bool,
                is_bitwise_serializable<
                    typename std::remove_const<Key>::type>::value &&
                    is_bitwise_serializable<
                        typename std::remove_const<Value>::type>::value>
        {
        };
    }    // namespace traits

    namespace serialization {

        namespace detail {

            template <typename Key, typename Value>
            void load_pair_impl(
                input_archive& ar, std::pair<Key, Value>& t, std::false_type)
            {
                ar >>
                    const_cast<typename std::add_lvalue_reference<
                        typename std::remove_const<Key>::type>::type>(t.first);
                ar >> t.second;
            }

            template <typename Key, typename Value>
            void load_pair_impl(
                input_archive& ar, std::pair<Key, Value>& t, std::true_type)
            {
                if (ar.disable_array_optimization())
                    load_pair_impl(ar, t, std::false_type());
                else
                    load_binary(ar, &t, sizeof(std::pair<Key, Value>));
            }

            template <typename Key, typename Value>
            void save_pair_impl(output_archive& ar,
                const std::pair<Key, Value>& t, std::false_type)
            {
                ar << t.first;
                ar << t.second;
            }

            template <typename Key, typename Value>
            void save_pair_impl(output_archive& ar,
                const std::pair<Key, Value>& t, std::true_type)
            {
                if (ar.disable_array_optimization())
                    save_pair_impl(ar, t, std::false_type());
                else
                    save_binary(ar, &t, sizeof(std::pair<Key, Value>));
            }

        }    // namespace detail

        template <typename Key, typename Value>
        void serialize(input_archive& ar, std::pair<Key, Value>& t, unsigned)
        {
            using pair_type = std::pair<Key, Value>;
            using optimized = std::integral_constant<bool,
                hpx::traits::is_bitwise_serializable<pair_type>::value>;

            detail::load_pair_impl(ar, t, optimized());
        }

        template <typename Key, typename Value>
        void serialize(
            output_archive& ar, const std::pair<Key, Value>& t, unsigned)
        {
            using pair_type = std::pair<Key, Value>;
            using optimized = std::integral_constant<bool,
                hpx::traits::is_bitwise_serializable<pair_type>::value>;

            detail::save_pair_impl(ar, t, optimized());
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
                t.insert(t.end(), std::move(v));
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
    }    // namespace serialization
}    // namespace hpx
