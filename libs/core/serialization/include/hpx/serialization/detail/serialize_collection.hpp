//  Copyright (c) 2017 Anton Bikineev
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/concepts/has_member_xxx.hpp>
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

    template <typename Container>
    HPX_FORCEINLINE void reserve_if_container(Container& v,
        std::size_t n) noexcept(!has_reserve_v<std::decay_t<Container>>)
    {
        if constexpr (has_reserve_v<std::decay_t<Container>>)
        {
            v.reserve(n);
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    template <typename Archive, typename Collection>
    void save_collection(Archive& ar, const Collection& collection)
    {
        using value_type = typename Collection::value_type;

        for (const auto& i : collection)
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
                collection.emplace_back(HPX_MOVE(elem));
            }
        }
        else
        {
            static constexpr bool is_polymorphic =
                hpx::traits::is_intrusive_polymorphic_v<value_type> ||
                hpx::traits::is_nonintrusive_polymorphic_v<value_type>;

            while (size-- != 0)
            {
                if constexpr (is_polymorphic)
                {
                    std::unique_ptr<value_type> data(
                        constructor_selector_ptr<value_type>::create(ar));
                    collection.emplace_back(HPX_MOVE(*data));
                }
                else
                {
                    collection.emplace_back(
                        constructor_selector<value_type>::create(ar));
                }
            }
        }
    }
}    // namespace hpx::serialization::detail
