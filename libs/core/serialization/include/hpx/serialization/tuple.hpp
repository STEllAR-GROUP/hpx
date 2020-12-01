//  Copyright (c) 2011-2013 Thomas Heller
//  Copyright (c) 2011-2013 Hartmut Kaiser
//  Copyright (c) 2013-2015 Agustin Berge
//  Copyright (c)      2019 Mikael Simberg
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/datastructures/tuple.hpp>
#include <hpx/serialization/detail/non_default_constructible.hpp>
#include <hpx/serialization/serialization_fwd.hpp>
#include <hpx/serialization/traits/is_bitwise_serializable.hpp>
#include <hpx/type_support/pack.hpp>

#include <cstddef>
#include <type_traits>

namespace hpx { namespace traits {

    template <typename... Ts>
    struct is_bitwise_serializable<::hpx::tuple<Ts...>>
      : ::hpx::util::all_of<hpx::traits::is_bitwise_serializable<
            typename std::remove_const<Ts>::type>...>
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
        static void call(Archive& ar, hpx::tuple<Ts...>& t, unsigned int)
        {
            int const _sequencer[] = {((ar & hpx::get<Is>(t)), 0)...};
            (void) _sequencer;
        }
    };

    template <typename Archive, std::size_t... Is, typename... Ts>
    struct load_construct_data_with_index_pack<Archive,
        hpx::util::index_pack<Is...>, Ts...>
    {
        static void call(
            Archive& ar, hpx::tuple<Ts...>& t, unsigned int version)
        {
            using serialization::detail::load_construct_data;
            int const _sequencer[] = {
                (load_construct_data(ar, &hpx::get<Is>(t), version), 0)...};
            (void) _sequencer;
        }
    };

    template <typename Archive, std::size_t... Is, typename... Ts>
    struct save_construct_data_with_index_pack<Archive,
        hpx::util::index_pack<Is...>, Ts...>
    {
        static void call(
            Archive& ar, hpx::tuple<Ts...> const& t, unsigned int version)
        {
            using serialization::detail::save_construct_data;
            int const _sequencer[] = {
                (save_construct_data(ar, &hpx::get<Is>(t), version), 0)...};
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
