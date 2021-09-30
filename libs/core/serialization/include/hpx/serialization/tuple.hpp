//  Copyright (c) 2011-2013 Thomas Heller
//  Copyright (c) 2011-2021 Hartmut Kaiser
//  Copyright (c) 2013-2015 Agustin Berge
//  Copyright (c)      2019 Mikael Simberg
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/datastructures/tuple.hpp>
#include <hpx/serialization/detail/constructor_selector.hpp>
#include <hpx/serialization/detail/non_default_constructible.hpp>
#include <hpx/serialization/detail/polymorphic_nonintrusive_factory.hpp>
#include <hpx/serialization/serialization_fwd.hpp>
#include <hpx/serialization/traits/is_bitwise_serializable.hpp>
#include <hpx/serialization/traits/is_not_bitwise_serializable.hpp>
#include <hpx/type_support/pack.hpp>

#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>

namespace hpx { namespace traits {

    template <typename... Ts>
    struct is_bitwise_serializable<::hpx::tuple<Ts...>>
      : ::hpx::util::all_of<hpx::traits::is_bitwise_serializable<
            typename std::remove_const<Ts>::type>...>
    {
    };

    template <typename... Ts>
    struct is_not_bitwise_serializable<::hpx::tuple<Ts...>>
      : std::integral_constant<bool,
            !is_bitwise_serializable_v<::hpx::tuple<Ts...>>>
    {
    };
}}    // namespace hpx::traits

namespace hpx { namespace util { namespace detail {

    template <typename Archive, typename Is, typename... Ts>
    struct serialize_with_index_pack;

    template <typename Archive, typename Is, typename... Ts>
    struct load_construct_data_with_index_pack;

    template <typename Archive, typename Is, typename... Ts>
    struct save_construct_data_with_index_pack;

    template <typename Archive, std::size_t... Is, typename... Ts>
    struct serialize_with_index_pack<Archive, hpx::util::index_pack<Is...>,
        Ts...>
    {
        template <typename T>
        static void call(Archive& ar, T& t, unsigned int)
        {
#if !defined(HPX_SERIALIZATION_HAVE_ALLOW_CONST_TUPLE_MEMBERS)
            int const _sequencer[] = {((ar & hpx::get<Is>(t)), 0)...};
#else
            int const _sequencer[] = {
                ((ar & const_cast<std::remove_const_t<Ts>&>(hpx::get<Is>(t))),
                    0)...};
#endif
            (void) _sequencer;
        }
    };

    template <typename Archive, std::size_t... Is, typename... Ts>
    struct load_construct_data_with_index_pack<Archive,
        hpx::util::index_pack<Is...>, Ts...>
    {
        template <typename T>
        static void load_element(Archive& ar, T& t)
        {
            using is_polymorphic = std::integral_constant<bool,
                hpx::traits::is_intrusive_polymorphic_v<T> ||
                    hpx::traits::is_nonintrusive_polymorphic_v<T>>;

            if constexpr (is_polymorphic::value)
            {
                std::unique_ptr<T> data(
                    serialization::detail::constructor_selector_ptr<T>::create(
                        ar));
                t = std::move(*data);
            }
            else
            {
                t = serialization::detail::constructor_selector<T>::create(ar);
            }
        }

        static void call(Archive& ar, hpx::tuple<Ts...>& t, unsigned int)
        {
#if !defined(HPX_SERIALIZATION_HAVE_ALLOW_CONST_TUPLE_MEMBERS)
            int const _sequencer[] = {
                (load_element(ar, hpx::get<Is>(t)), 0)...};
#else
            int const _sequencer[] = {
                (load_element(
                     ar, const_cast<std::remove_const_t<Ts>&>(hpx::get<Is>(t))),
                    0)...};
#endif

            (void) _sequencer;
        }
    };

    template <typename Archive, std::size_t... Is, typename... Ts>
    struct save_construct_data_with_index_pack<Archive,
        hpx::util::index_pack<Is...>, Ts...>
    {
        template <typename T>
        static void save_element(Archive& ar, T& t)
        {
            if constexpr (!std::is_default_constructible<T>::value)
            {
                using serialization::detail::save_construct_data;
                save_construct_data(ar, &t, 0);
            }
            ar << t;
        }

        static void call(Archive& ar, hpx::tuple<Ts...> const& t, unsigned int)
        {
            int const _sequencer[] = {
                (save_element(ar, hpx::get<Is>(t)), 0)...};
            (void) _sequencer;
        }
    };
}}}    // namespace hpx::util::detail

namespace hpx { namespace serialization {

    template <typename Archive, typename... Ts>
    void serialize(Archive& ar, hpx::tuple<Ts...>& t, unsigned int version)
    {
        using Is = typename hpx::util::make_index_pack<sizeof...(Ts)>::type;
        hpx::util::detail::serialize_with_index_pack<Archive, Is, Ts...>::call(
            ar, t, version);
    }

    template <typename Archive>
    void serialize(Archive&, hpx::tuple<>&, unsigned)
    {
    }

    template <typename Archive, typename... Ts>
    void load_construct_data(
        Archive& ar, hpx::tuple<Ts...>* t, unsigned int version)
    {
        using Is = typename hpx::util::make_index_pack<sizeof...(Ts)>::type;
        hpx::util::detail::load_construct_data_with_index_pack<Archive, Is,
            Ts...>::call(ar, *t, version);
    }

    template <typename Archive, typename... Ts>
    void save_construct_data(
        Archive& ar, hpx::tuple<Ts...> const* t, unsigned int version)
    {
        using Is = typename hpx::util::make_index_pack<sizeof...(Ts)>::type;
        hpx::util::detail::save_construct_data_with_index_pack<Archive, Is,
            Ts...>::call(ar, *t, version);
    }
}}    // namespace hpx::serialization
