//  Copyright (c) 2011-2013 Thomas Heller
//  Copyright (c) 2013-2015 Agustin Berge
//  Copyright (c) 2019 Mikael Simberg
//  Copyright (c) 2020-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/serialization/serialization_fwd.hpp>
#include <hpx/serialization/traits/is_bitwise_serializable.hpp>
#include <hpx/serialization/traits/is_not_bitwise_serializable.hpp>
#include <hpx/type_support/pack.hpp>

#include <cstddef>
#include <tuple>
#include <type_traits>

namespace hpx { namespace traits {

    template <typename... Ts>
    struct is_bitwise_serializable<std::tuple<Ts...>>
      : ::hpx::util::all_of<hpx::traits::is_bitwise_serializable<
            typename std::remove_const<Ts>::type>...>
    {
    };

    template <typename... Ts>
    struct is_not_bitwise_serializable<std::tuple<Ts...>>
      : std::integral_constant<bool,
            !is_bitwise_serializable_v<std::tuple<Ts...>>>
    {
    };
}}    // namespace hpx::traits

namespace hpx { namespace serialization {

    namespace detail {

        template <typename Archive, typename Is, typename... Ts>
        struct std_serialize_with_index_pack;

        template <typename Archive, std::size_t... Is, typename... Ts>
        struct std_serialize_with_index_pack<Archive,
            hpx::util::index_pack<Is...>, Ts...>
        {
            static void call(Archive& ar, std::tuple<Ts...>& t, unsigned int)
            {
#if !defined(HPX_SERIALIZATION_HAVE_ALLOW_CONST_TUPLE_MEMBERS)
                int const _sequencer[] = {((ar & std::get<Is>(t)), 0)...};
#else
                int const _sequencer[] = {
                    ((ar &
                         const_cast<std::remove_const_t<Ts>&>(std::get<Is>(t))),
                        0)...};
#endif
                (void) _sequencer;
            }
        };
    }    // namespace detail

    template <typename Archive, typename... Ts>
    void serialize(Archive& ar, std::tuple<Ts...>& t, unsigned int version)
    {
        using Is = typename hpx::util::make_index_pack<sizeof...(Ts)>::type;
        detail::std_serialize_with_index_pack<Archive, Is, Ts...>::call(
            ar, t, version);
    }

    template <typename Archive>
    void serialize(Archive&, std::tuple<>&, unsigned int)
    {
    }

}}    // namespace hpx::serialization
