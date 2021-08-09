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

namespace hpx { namespace serialization { namespace detail {

    ////////////////////////////////////////////////////////////////////////////
    // not every random access sequence is reservable, so we need an explicit
    // trait to determine this
    HPX_HAS_MEMBER_XXX_TRAIT_DEF(reserve)

    template <typename Collection>
    using is_reservable = std::integral_constant<bool,
        has_reserve<typename std::decay<Collection>::type>::value>;

    ////////////////////////////////////////////////////////////////////////////
    template <typename Container>
    HPX_FORCEINLINE
        typename std::enable_if<!is_reservable<Container>::value>::type
        reserve_if_container(Container&, std::size_t) noexcept
    {
    }

    template <typename Container>
    HPX_FORCEINLINE
        typename std::enable_if<is_reservable<Container>::value>::type
        reserve_if_container(Container& v, std::size_t n)
    {
        v.reserve(n);
    }

    ////////////////////////////////////////////////////////////////////////////
    template <typename Value>
    class save_collection_impl
    {
        struct default_
        {
            template <typename Archive, typename Collection>
            static void call(Archive& ar, const Collection& collection)
            {
                for (const auto& i : collection)
                    ar << i;
            }
        };

        struct non_default_
        {
            template <typename Archive, typename Collection>
            static void call(Archive& ar, const Collection& collection)
            {
                // for non-default constructible types we also provide
                // a customization point to save any external data
                // (in the same way as Boost.Serialization does)
                for (const auto& i : collection)
                {
                    save_construct_data(ar, &i, 0);
                    ar << i;
                }
            }
        };

    public:
        using type =
            std::conditional_t<std::is_default_constructible<Value>::value,
                default_, non_default_>;
    };

    template <typename Value>
    class load_collection_impl
    {
        struct default_
        {
            template <typename Archive, typename Collection>
            static void call(Archive& ar, Collection& collection,
                typename Collection::size_type size)
            {
                using value_type = typename Collection::value_type;

                collection.clear();
                reserve_if_container(collection, size);

                while (size-- != 0)
                {
                    value_type elem;
                    ar >> elem;
                    collection.emplace_back(std::move(elem));
                }
            }
        };

        struct non_default_
        {
            template <typename Archive, typename Collection>
            static void call(Archive& ar, Collection& collection,
                typename Collection::size_type size)
            {
                using value_type = typename Collection::value_type;

                using is_polymorphic = std::integral_constant<bool,
                    hpx::traits::is_intrusive_polymorphic_v<value_type> ||
                        hpx::traits::is_nonintrusive_polymorphic_v<value_type>>;

                call(ar, collection, size, is_polymorphic());
            }

            template <typename Archive, typename Collection>
            static void call(Archive& ar, Collection& collection,
                typename Collection::size_type size, std::false_type)
            {
                using value_type = typename Collection::value_type;

                collection.clear();
                reserve_if_container(collection, size);

                while (size-- > 0)
                {
                    collection.emplace_back(
                        constructor_selector<value_type>::create(ar));
                }
            }

            template <typename Archive, typename Collection>
            static void call(Archive& ar, Collection& collection,
                typename Collection::size_type size, std::true_type)
            {
                using value_type = typename Collection::value_type;

                collection.clear();
                reserve_if_container(collection, size);

                while (size-- > 0)
                {
                    std::unique_ptr<value_type> data(
                        constructor_selector_ptr<value_type>::create(ar));
                    collection.emplace_back(std::move(*data));
                }
            }
        };

    public:
        using type =
            std::conditional_t<std::is_default_constructible<Value>::value,
                default_, non_default_>;
    };

    template <typename Archive, typename Collection>
    void save_collection(Archive& ar, const Collection& collection)
    {
        using value_type = typename Collection::value_type;
        save_collection_impl<value_type>::type::call(ar, collection);
    }

    template <typename Archive, typename Collection>
    void load_collection(Archive& ar, Collection& collection,
        typename Collection::size_type size)
    {
        using value_type = typename Collection::value_type;
        load_collection_impl<value_type>::type::call(ar, collection, size);
    }

}}}    // namespace hpx::serialization::detail
