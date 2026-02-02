//  Copyright (c) 2017 Anton Bikineev
//  Copyright (c) 2022-2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/modules/concepts.hpp>
#include <hpx/serialization/detail/constructor_selector.hpp>
#include <hpx/serialization/detail/polymorphic_nonintrusive_factory.hpp>
#include <hpx/serialization/serialization_fwd.hpp>

#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>

namespace hpx::serialization::detail {

    ////////////////////////////////////////////////////////////////////////////
    // not every random access sequence is reservable, so we need an explicit
    // trait to determine this
    HPX_HAS_MEMBER_XXX_TRAIT_DEF(reserve)
    HPX_HAS_MEMBER_XXX_TRAIT_DEF(emplace_back)
    HPX_HAS_MEMBER_XXX_TRAIT_DEF(emplace)

    template <typename Container>
    HPX_FORCEINLINE void reserve_if_container(
        Container& v, std::size_t n) noexcept(!has_reserve_v<Container>)
    {
        if constexpr (has_reserve_v<Container>)
        {
            v.reserve(n);
        }
    }

    template <typename Container, typename T>
    void emplace_into_collection(Container& c, T&& t)
    {
        if constexpr (has_emplace_back_v<Container>)
        {    // vectors, lists, etc.
            c.emplace_back(HPX_FORWARD(T, t));
        }
        else if constexpr (has_emplace_v<Container>)
        {    // sets, maps, etc.
            c.emplace(HPX_FORWARD(T, t));
        }
        else
        {
            c.insert(c.end(), HPX_FORWARD(T, t));
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    template <typename Archive, typename Collection>
    void save_collection(Archive& ar, Collection const& collection)
    {
        using value_type = typename Collection::value_type;

        for (auto const& i : collection)
        {
            if constexpr (!std::is_default_constructible_v<value_type>)
            {
                // for non-default constructible types we also provide
                // a customization point to save any external data
                // (in the same way as Boost.Serialization does)
                save_construct_data(ar, &i, 0);
            }
            ar << i;
        }
    }

    template <typename Archive, typename Collection>
    void load_collection(Archive& ar, Collection& collection,
        typename Collection::size_type size)
    {
        using value_type = typename Collection::value_type;

        collection.clear();
        reserve_if_container(collection, size);

        if constexpr (std::is_default_constructible_v<value_type>)
        {
            while (size-- != 0)
            {
                value_type elem;
                ar >> elem;
                emplace_into_collection(collection, HPX_MOVE(elem));
            }
        }
        else
        {
            constexpr bool is_polymorphic =
                hpx::traits::is_intrusive_polymorphic_v<value_type> ||
                hpx::traits::is_nonintrusive_polymorphic_v<value_type>;

            while (size-- != 0)
            {
                if constexpr (is_polymorphic)
                {
                    std::unique_ptr<value_type> data(
                        constructor_selector_ptr<value_type>::create(ar));
                    emplace_into_collection(collection, HPX_MOVE(*data));
                }
                else
                {
                    emplace_into_collection(collection,
                        constructor_selector<value_type>::create(ar));
                }
            }
        }
    }
}    // namespace hpx::serialization::detail
