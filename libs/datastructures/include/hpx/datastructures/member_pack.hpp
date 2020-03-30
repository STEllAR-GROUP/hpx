//  Copyright (c) 2020 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/type_support/pack.hpp>

#include <cstddef>    // for size_t
#include <type_traits>
#include <utility>

namespace hpx { namespace util {

    namespace detail {

        template <std::size_t I, typename T,
            bool Empty = std::is_empty<T>::value && !std::is_final<T>::value>
        struct member_leaf
        {
            T member;

            member_leaf() = default;

            template <typename U>
            explicit constexpr member_leaf(std::piecewise_construct_t, U&& v)
              : member(std::forward<U>(v))
            {
            }
        };

        template <std::size_t I, typename T>
        struct member_leaf<I, T, /*Empty*/ true> : T
        {
            member_leaf() = default;

            template <typename U>
            explicit constexpr member_leaf(std::piecewise_construct_t, U&& v)
              : T(std::forward<U>(v))
            {
            }
        };

        template <std::size_t I, typename T>
        T member_type(member_leaf<I, T> const& /*leaf*/) noexcept;

        template <std::size_t I, typename T>
        static constexpr T& member_get(member_leaf<I, T, false>& leaf) noexcept
        {
            return leaf.member;
        }
        template <std::size_t I, typename T>
        static constexpr T& member_get(member_leaf<I, T, true>& leaf) noexcept
        {
            return leaf;
        }
        template <std::size_t I, typename T>
        static constexpr T const& member_get(
            member_leaf<I, T, false> const& leaf) noexcept
        {
            return leaf.member;
        }
        template <std::size_t I, typename T>
        static constexpr T const& member_get(
            member_leaf<I, T, true> const& leaf) noexcept
        {
            return leaf;
        }

    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////
    template <typename Is, typename... Ts>
    struct HPX_EMPTY_BASES member_pack;

    template <std::size_t... Is, typename... Ts>
    struct HPX_EMPTY_BASES member_pack<util::index_pack<Is...>, Ts...>
      : detail::member_leaf<Is, Ts>...
    {
        member_pack() = default;

        template <typename... Us>
        explicit constexpr member_pack(std::piecewise_construct_t, Us&&... us)
          : detail::member_leaf<Is, Ts>(
                std::piecewise_construct, std::forward<Us>(us))...
        {
        }

        template <std::size_t I>
            constexpr decltype(auto) get() & noexcept
        {
            return detail::member_get<I>(*this);
        }
        template <std::size_t I>
        constexpr decltype(auto) get() const& noexcept
        {
            return detail::member_get<I>(*this);
        }
        template <std::size_t I>
            constexpr decltype(auto) get() && noexcept
        {
            using T = decltype(detail::member_type<I>(*this));
            return std::forward<T>(detail::member_get<I>(*this));
        }
        template <std::size_t I>
        constexpr decltype(auto) get() const&& noexcept
        {
            using T = decltype(detail::member_type<I>(*this));
            return std::forward<T>(detail::member_get<I>(*this));
        }
    };

    template <typename... Ts>
    using member_pack_for =
        member_pack<typename util::make_index_pack<sizeof...(Ts)>::type, Ts...>;

}}    // namespace hpx::util

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace serialization {
    ///////////////////////////////////////////////////////////////////////////
    template <typename Archive, std::size_t... Is, typename... Ts>
    HPX_FORCEINLINE void serialize(Archive& ar,
        ::hpx::util::member_pack<util::index_pack<Is...>, Ts...>& mp,
        unsigned int const /*version*/ = 0)
    {
        int sequencer[] = {((ar & mp.template get<Is>()), 0)...};
        (void) sequencer;
    }
}}    // namespace hpx::serialization
