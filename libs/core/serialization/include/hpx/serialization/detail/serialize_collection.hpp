//  Copyright (c) 2017 Anton Bikineev
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/concepts/has_member_xxx.hpp>
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
        using type = typename std::conditional<
            std::is_default_constructible<Value>::value, default_,
            non_default_>::type;
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
                collection.resize(size);
                for (auto& i : collection)
                    ar >> i;
            }
        };

        struct non_default_
        {
            template <typename Archive, typename Collection>
            static void call(Archive& ar, Collection& collection,
                typename Collection::size_type size)
            {
                using value_type = typename Collection::value_type;

                collection.clear();
                reserve_if_container(collection, size);

                while (size-- > 0)
                {
                    std::unique_ptr<value_type> data(
                        constructor_selector<value_type>::create(ar));
                    collection.push_back(std::move(*data));
                }
            }
        };

    public:
        using type = typename std::conditional<
            std::is_default_constructible<Value>::value, default_,
            non_default_>::type;
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
