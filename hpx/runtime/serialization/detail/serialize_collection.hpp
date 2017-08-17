//  Copyright (c) 2017 Anton Bikineev
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_SERIALIZATION_DETAIL_SERIALIZE_COLLECTION_HPP
#define HPX_SERIALIZATION_DETAIL_SERIALIZE_COLLECTION_HPP

#include <hpx/config.hpp>
#include <hpx/traits/detail/reserve.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace serialization { namespace detail {

    // default fallbacks
    template <class Archive, class T>
    void save_construct_data(Archive&, T*, unsigned)
    {
        // a user is not required to provide their own adl-overload
    }

    template <class Archive, class T>
    void load_construct_data(Archive&, T* t, unsigned)
    {
        // this function is never supposed to be called
        HPX_ASSERT(false);
        ::new (t) T;
    }

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
                using storage_type = std::aligned_storage<
                    sizeof(value_type), alignof(value_type)>;

                collection.clear();
                hpx::traits::detail::reserve_if_reservable(collection, size);

                while (size-- > 0)
                {
                    storage_type storage;
                    value_type& ref = reinterpret_cast<value_type&>(storage);
                    load_construct_data(ar, &ref, 0);
                    ar >> ref;
                    collection.push_back(std::move(ref));
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
