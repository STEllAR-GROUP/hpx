//  Copyright (c) 2011-2013 Thomas Heller
//  Copyright (c) 2011-2013 Hartmut Kaiser
//  Copyright (c) 2013-2015 Agustin Berge
//  Copyright (c)      2019 Mikael Simberg
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_SERIALIZATION_TUPLE_HPP)
#define HPX_RUNTIME_SERIALIZATION_TUPLE_HPP

#include <hpx/runtime/serialization/detail/non_default_constructible.hpp>
#include <hpx/traits/is_bitwise_serializable.hpp>
#include <hpx/datastructures/detail/pack.hpp>
#include <hpx/datastructures/tuple.hpp>

#include <cstddef>
#include <type_traits>

namespace hpx { namespace traits {
    template <typename... Ts>
    struct is_bitwise_serializable<::hpx::util::tuple<Ts...>>
      : ::hpx::util::detail::all_of<hpx::traits::is_bitwise_serializable<
            typename std::remove_const<Ts>::type>...>
    {
    };
}}

namespace hpx { namespace util { namespace detail {
    template <typename Archive, typename Is, typename... Ts>
    struct serialize_with_index_pack;

    template <typename Archive, typename Is, typename... Ts>
    struct load_construct_data_with_index_pack;

    template <typename Archive, typename Is, typename... Ts>
    struct save_construct_data_with_index_pack;

    template <typename Archive, std::size_t... Is, typename... Ts>
    struct serialize_with_index_pack<Archive,
        hpx::util::detail::pack_c<std::size_t, Is...>, Ts...>
    {
        static void call(Archive& ar, hpx::util::tuple<Ts...>& t, unsigned int)
        {
            int const _sequencer[] = {((ar & hpx::util::get<Is>(t)), 0)...};
            (void) _sequencer;
        }
    };

    template <typename Archive, std::size_t... Is, typename... Ts>
    struct load_construct_data_with_index_pack<Archive,
        hpx::util::detail::pack_c<std::size_t, Is...>, Ts...>
    {
        static void call(
            Archive& ar, hpx::util::tuple<Ts...>& t, unsigned int version)
        {
            using serialization::detail::load_construct_data;
            int const _sequencer[] = {
                (load_construct_data(ar, &hpx::util::get<Is>(t), version),
                    0)...};
            (void) _sequencer;
        }
    };

    template <typename Archive, std::size_t... Is, typename... Ts>
    struct save_construct_data_with_index_pack<Archive,
        hpx::util::detail::pack_c<std::size_t, Is...>, Ts...>
    {
        static void call(
            Archive& ar, hpx::util::tuple<Ts...> const& t, unsigned int version)
        {
            using serialization::detail::save_construct_data;
            int const _sequencer[] = {
                (save_construct_data(ar, &hpx::util::get<Is>(t), version),
                    0)...};
            (void) _sequencer;
        }
    };
}}}

namespace hpx { namespace serialization {
    template <typename Archive, typename... Ts>
    void serialize(
        Archive& ar, hpx::util::tuple<Ts...>& t, unsigned int version)
    {
        using Is =
            typename hpx::util::detail::make_index_pack<sizeof...(Ts)>::type;
        hpx::util::detail::serialize_with_index_pack<Archive, Is, Ts...>::call(
            ar, t, version);
    }

    template <typename Archive>
    void serialize(Archive& ar, hpx::util::tuple<>&, unsigned)
    {
    }

    template <typename Archive, typename... Ts>
    void load_construct_data(
        Archive& ar, hpx::util::tuple<Ts...>* t, unsigned int version)
    {
        using Is =
            typename hpx::util::detail::make_index_pack<sizeof...(Ts)>::type;
        hpx::util::detail::load_construct_data_with_index_pack<Archive, Is,
            Ts...>::call(ar, *t, version);
    }

    template <typename Archive, typename... Ts>
    void save_construct_data(
        Archive& ar, hpx::util::tuple<Ts...> const* t, unsigned int version)
    {
        using Is =
            typename hpx::util::detail::make_index_pack<sizeof...(Ts)>::type;
        hpx::util::detail::save_construct_data_with_index_pack<Archive, Is,
            Ts...>::call(ar, *t, version);
    }
}}

#endif
