//  Copyright (c) 2017 Anton Bikineev
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_SERIALIZATION_DETAIL_SERIALIZE_COLLECTION_HPP
#define HPX_SERIALIZATION_DETAIL_SERIALIZE_COLLECTION_HPP

#include <hpx/config.hpp>
#include <hpx/traits/detail/reserve.hpp>
#include <hpx/runtime/serialization/detail/polymorphic_nonintrusive_factory.hpp>

#include <memory>
#include <type_traits>
#include <utility>

namespace hpx { namespace serialization { namespace detail
{
    template <class Value>
    class save_collection_impl
    {
        struct default_
        {
            template <class Archive, class Collection>
            static void call(Archive& ar, const Collection& collection)
            {
                for (const auto& i: collection)
                    ar << i;
            }
        };

        struct non_default_
        {
            template <class Archive, class Collection>
            static void call(Archive& ar, const Collection& collection)
            {
                // for non-default constructible types we also provide
                // a customization point to save any external data
                // (in the same way as Boost.Serialization does)
                for (const auto& i: collection)
                {
                    save_construct_data(ar, &i, 0);
                    ar << i;
                }
            }
        };

    public:
        using type = typename std::conditional<
            std::is_default_constructible<Value>::value,
                default_, non_default_>::type;
    };

    template <class Value>
    class load_collection_impl
    {
        struct default_
        {
            template <class Archive, class Collection>
            static void call(Archive& ar, Collection& collection,
                    typename Collection::size_type size)
            {
                collection.resize(size);
                for (auto& i: collection)
                    ar >> i;
            }
        };

        struct non_default_
        {
            template <class Archive, class Collection>
            static void call(Archive& ar, Collection& collection,
                    typename Collection::size_type size)
            {
                using value_type = typename Collection::value_type;

                collection.clear();
                hpx::traits::detail::reserve_if_reservable(collection, size);

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
            std::is_default_constructible<Value>::value,
                default_, non_default_>::type;
    };

    template <class Archive, class Collection>
    void save_collection(Archive& ar, const Collection& collection)
    {
        using value_type = typename Collection::value_type;
        save_collection_impl<value_type>::type::call(ar, collection);
    }

    template <class Archive, class Collection>
    void load_collection(Archive& ar, Collection& collection,
            typename Collection::size_type size)
    {
        using value_type = typename Collection::value_type;
        load_collection_impl<value_type>::type::call(ar, collection, size);
    }

}}} // hpx::serialization::detail

#endif // HPX_SERIALIZATION_DETAIL_SERIALIZE_COLLECTION_HPP
