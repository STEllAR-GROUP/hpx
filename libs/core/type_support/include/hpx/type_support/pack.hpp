//  Copyright (c) 2014-2020 Agustin Berge
//  Copyright (c) 2024-2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <cstddef>
#include <type_traits>

namespace hpx::util {

    HPX_CORE_MODULE_EXPORT_EXTERN template <typename... Ts>
    struct pack
    {
        using type = pack;
        static constexpr std::size_t size = sizeof...(Ts);
    };

    HPX_CORE_MODULE_EXPORT_EXTERN template <typename T, T... Vs>
    struct pack_c
    {
        using type = pack_c;
        static constexpr std::size_t size = sizeof...(Vs);
    };

    HPX_CORE_MODULE_EXPORT_EXTERN template <std::size_t... Is>
    using index_pack = pack_c<std::size_t, Is...>;

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {

        HPX_CORE_MODULE_EXPORT_EXTERN template <typename Left, typename Right>
        struct make_index_pack_join;

        HPX_CORE_MODULE_EXPORT_EXTERN template <std::size_t... Left,
            std::size_t... Right>
        struct make_index_pack_join<index_pack<Left...>, index_pack<Right...>>
          : index_pack<Left..., (sizeof...(Left) + Right)...>
        {
        };
    }    // namespace detail

    HPX_CORE_MODULE_EXPORT_EXTERN template <std::size_t N>
    struct make_index_pack
      : detail::make_index_pack_join<typename make_index_pack<N / 2>::type,
            typename make_index_pack<N - N / 2>::type>
    {
    };

    template <>
    struct make_index_pack<0> : pack_c<std::size_t>
    {
    };

    template <>
    struct make_index_pack<1> : index_pack<0>
    {
    };

    HPX_CORE_MODULE_EXPORT_EXTERN template <std::size_t N>
    using make_index_pack_t = typename make_index_pack<N>::type;

    ///////////////////////////////////////////////////////////////////////////
    // Workaround for clang bug [https://bugs.llvm.org/show_bug.cgi?id=35077]
    namespace detail {

        HPX_CORE_MODULE_EXPORT_EXTERN template <typename T>
        struct is_true
          : std::integral_constant<bool, static_cast<bool>(T::value)>
        {
        };

        HPX_CORE_MODULE_EXPORT_EXTERN template <typename T>
        struct is_false
          : std::integral_constant<bool, !static_cast<bool>(T::value)>
        {
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {

        HPX_CORE_MODULE_EXPORT_EXTERN template <typename... Ts>
        struct always_true : std::true_type
        {
        };

        HPX_CORE_MODULE_EXPORT_EXTERN template <typename... Ts>
        struct always_false : std::false_type
        {
        };

        HPX_CORE_MODULE_EXPORT_EXTERN template <typename... Ts>
        constexpr std::false_type all_of(...) noexcept;

        HPX_CORE_MODULE_EXPORT_EXTERN template <typename... Ts>
        constexpr auto all_of(int) noexcept
            -> always_true<std::enable_if_t<is_true<Ts>::value>...>;
    }    // namespace detail

    HPX_CORE_MODULE_EXPORT_EXTERN template <typename... Ts>
    struct all_of : decltype(detail::all_of<Ts...>(0))
    {
    };

    template <>
    struct all_of<>    // <fake-type>
      : std::true_type
    {
    };

    HPX_CORE_MODULE_EXPORT_EXTERN template <typename... Ts>
    inline constexpr bool all_of_v = all_of<Ts...>::value;

    namespace detail {

        HPX_CORE_MODULE_EXPORT_EXTERN template <typename... Ts>
        constexpr std::true_type any_of(...) noexcept;

        HPX_CORE_MODULE_EXPORT_EXTERN template <typename... Ts>
        constexpr auto any_of(int) noexcept
            -> always_false<std::enable_if_t<is_false<Ts>::value>...>;
    }    // namespace detail

    HPX_CORE_MODULE_EXPORT_EXTERN template <typename... Ts>
    struct any_of : decltype(detail::any_of<Ts...>(0))
    {
    };

    template <>
    struct any_of<>    // <fake-type>
      : std::false_type
    {
    };

    HPX_CORE_MODULE_EXPORT_EXTERN template <typename... Ts>
    inline constexpr bool any_of_v = any_of<Ts...>::value;

    HPX_CORE_MODULE_EXPORT_EXTERN template <typename... Ts>
    struct none_of : std::integral_constant<bool, !any_of<Ts...>::value>
    {
    };

    HPX_CORE_MODULE_EXPORT_EXTERN template <typename... Ts>
    inline constexpr bool none_of_v = none_of<Ts...>::value;

    HPX_CORE_MODULE_EXPORT_EXTERN template <typename T, typename... Ts>
    struct contains : any_of<std::is_same<T, Ts>...>
    {
    };

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {

        HPX_CORE_MODULE_EXPORT_EXTERN struct empty_helper
        {
        };

        HPX_CORE_MODULE_EXPORT_EXTERN template <std::size_t I, typename T>
        struct indexed
        {
            using type = T;
        };

        HPX_CORE_MODULE_EXPORT_EXTERN template <typename Ts, typename Is>
        struct indexer;

        HPX_CORE_MODULE_EXPORT_EXTERN template <typename... Ts,
            std::size_t... Is>
        struct indexer<pack<Ts...>, pack_c<std::size_t, Is...>>
          : indexed<Is, Ts>...
        {
        };

        HPX_CORE_MODULE_EXPORT_EXTERN template <std::size_t J>
        constexpr empty_helper at_index_check(...) noexcept;

        HPX_CORE_MODULE_EXPORT_EXTERN template <std::size_t J, typename T>
        constexpr indexed<J, T> at_index_check(indexed<J, T> const&) noexcept;

        HPX_CORE_MODULE_EXPORT_EXTERN template <std::size_t I, typename Ts>
        struct at_index_impl
          : decltype(detail::at_index_check<I>(
                indexer<Ts, make_index_pack_t<Ts::size>>()))
        {
        };
    }    // namespace detail

    HPX_CORE_MODULE_EXPORT_EXTERN template <std::size_t I, typename... Ts>
    struct at_index : detail::at_index_impl<I, pack<Ts...>>
    {
    };

    HPX_CORE_MODULE_EXPORT_EXTERN template <std::size_t I, typename... Ts>
    using at_index_t = typename at_index<I, Ts...>::type;

    namespace detail {

        HPX_CORE_MODULE_EXPORT_EXTERN template <typename Pack,
            template <typename> class Transformer>
        struct transform;

        HPX_CORE_MODULE_EXPORT_EXTERN template <
            template <typename> class Transformer,
            template <typename...> class Pack, typename... Ts>
        struct transform<Pack<Ts...>, Transformer>
        {
            using type = Pack<typename Transformer<Ts>::type...>;
        };

        /// Apply a meta-function to each element in a pack.
        HPX_CORE_MODULE_EXPORT_EXTERN template <typename Pack,
            template <typename> class Transformer>
        using transform_t = typename transform<Pack, Transformer>::type;

        HPX_CORE_MODULE_EXPORT_EXTERN template <typename PackUnique,
            typename PackRest>
        struct unique_helper;

        HPX_CORE_MODULE_EXPORT_EXTERN template <
            template <typename...> class Pack, typename... Ts>
        struct unique_helper<Pack<Ts...>, Pack<>>
        {
            using type = Pack<Ts...>;
        };

        HPX_CORE_MODULE_EXPORT_EXTERN template <
            template <typename...> class Pack, typename... Ts, typename U,
            typename... Us>
        struct unique_helper<Pack<Ts...>, Pack<U, Us...>>
          : std::conditional_t<contains<U, Ts...>::value,
                unique_helper<Pack<Ts...>, Pack<Us...>>,
                unique_helper<Pack<Ts..., U>, Pack<Us...>>>
        {
        };

        HPX_CORE_MODULE_EXPORT_EXTERN template <typename Pack>
        struct unique;

        HPX_CORE_MODULE_EXPORT_EXTERN template <
            template <typename...> class Pack, typename... Ts>
        struct unique<Pack<Ts...>> : unique_helper<Pack<>, Pack<Ts...>>
        {
        };

        /// Remove duplicate types in the given pack.
        HPX_CORE_MODULE_EXPORT_EXTERN template <typename Pack>
        using unique_t = typename unique<Pack>::type;

        HPX_CORE_MODULE_EXPORT_EXTERN template <typename... Packs>
        struct concat;

        HPX_CORE_MODULE_EXPORT_EXTERN template <
            template <typename...> class Pack, typename... Ts>
        struct concat<Pack<Ts...>>
        {
            using type = Pack<Ts...>;
        };

        HPX_CORE_MODULE_EXPORT_EXTERN template <
            template <typename...> class Pack, typename... Ts, typename... Us,
            typename... Rest>
        struct concat<Pack<Ts...>, Pack<Us...>, Rest...>
          : concat<Pack<Ts..., Us...>, Rest...>
        {
        };

        /// Concatenate the elements in the given packs into a single pack. The
        /// packs must be of the same type.
        HPX_CORE_MODULE_EXPORT_EXTERN template <typename... Packs>
        using concat_t = typename concat<Packs...>::type;

        /// Concatenate the elements in the given packs into a single pack and then
        /// remove duplicates.
        HPX_CORE_MODULE_EXPORT_EXTERN template <typename... Packs>
        using unique_concat_t = unique_t<concat_t<Packs...>>;

        HPX_CORE_MODULE_EXPORT_EXTERN template <typename Pack>
        struct concat_pack_of_packs;

        HPX_CORE_MODULE_EXPORT_EXTERN template <
            template <typename...> class Pack, typename... Ts>
        struct concat_pack_of_packs<Pack<Ts...>>
        {
            using type = typename concat<Ts...>::type;
        };

        /// Concatenate the packs in the given pack into a single pack. The
        /// outer pack is discarded.
        HPX_CORE_MODULE_EXPORT_EXTERN template <typename Pack>
        using concat_pack_of_packs_t =
            typename concat_pack_of_packs<Pack>::type;

        HPX_CORE_MODULE_EXPORT_EXTERN template <typename Pack>
        struct concat_inner_packs;

        HPX_CORE_MODULE_EXPORT_EXTERN template <
            template <typename...> class Pack>
        struct concat_inner_packs<Pack<>>
        {
            using type = Pack<>;
        };

        HPX_CORE_MODULE_EXPORT_EXTERN template <
            template <typename...> class Pack, typename T, typename... Ts>
        struct concat_inner_packs<Pack<T, Ts...>>
        {
            using type = Pack<typename concat<T, Ts...>::type>;
        };

        /// Concatenate the packs in the given pack into a single pack. The
        /// outer pack is kept.
        HPX_CORE_MODULE_EXPORT_EXTERN template <typename Pack>
        using concat_inner_packs_t = typename concat_inner_packs<Pack>::type;

        HPX_CORE_MODULE_EXPORT_EXTERN template <typename Pack, typename T>
        struct prepend;

        HPX_CORE_MODULE_EXPORT_EXTERN template <typename T,
            template <typename...> class Pack, typename... Ts>
        struct prepend<Pack<Ts...>, T>
        {
            using type = Pack<T, Ts...>;
        };

        /// Prepend a given type to the given pack.
        HPX_CORE_MODULE_EXPORT_EXTERN template <typename Pack, typename T>
        using prepend_t = typename prepend<Pack, T>::type;

        HPX_CORE_MODULE_EXPORT_EXTERN template <typename Pack, typename T>
        struct append;

        HPX_CORE_MODULE_EXPORT_EXTERN template <typename T,
            template <typename...> class Pack, typename... Ts>
        struct append<Pack<Ts...>, T>
        {
            using type = Pack<Ts..., T>;
        };

        /// Append a given type to the given pack.
        HPX_CORE_MODULE_EXPORT_EXTERN template <typename Pack, typename T>
        using append_t = typename append<Pack, T>::type;

        HPX_CORE_MODULE_EXPORT_EXTERN template <
            template <typename...> class NewPack, typename OldPack>
        struct change_pack;

        HPX_CORE_MODULE_EXPORT_EXTERN template <
            template <typename...> class NewPack,
            template <typename...> class OldPack, typename... Ts>
        struct change_pack<NewPack, OldPack<Ts...>>
        {
            using type = NewPack<Ts...>;
        };

        /// Change a OldPack<Ts...> to NewPack<Ts...>
        HPX_CORE_MODULE_EXPORT_EXTERN template <
            template <typename...> class NewPack, typename OldPack>
        using change_pack_t = typename change_pack<NewPack, OldPack>::type;
    }    // namespace detail
}    // namespace hpx::util
